
import os, re
import numpy as np
from skimage import io, filters, morphology, measure, segmentation, exposure, util
from skimage.restoration import rolling_ball
from scipy import ndimage as ndi  # for binary_fill_holes

def _to_gray(arr):
    if arr.ndim == 3:
        return util.img_as_float(arr.mean(axis=-1))
    return util.img_as_float(arr)

def background_subtract(img, radius_px=50):
    try:
        bg = rolling_ball(img, radius=radius_px)
        out = img - bg
        out[out<0]=0
        return out
    except Exception:
        return img

def autoseg_from_phase(img):
    blur = filters.gaussian(img, sigma=1.0)
    th = filters.threshold_otsu(blur)
    bw = blur > th
    bw = morphology.remove_small_objects(bw, min_size=2000)
    bw = morphology.binary_closing(bw, morphology.disk(5))
    # replace morphology.binary_fill_holes -> ndi.binary_fill_holes
    bw = ndi.binary_fill_holes(bw)
    return bw

def rim_core(mask, rim_px=20):
    er = morphology.erosion(mask, morphology.disk(rim_px))
    rim = mask ^ er
    core = er
    return rim, core

def radial_profile(image, mask, n_bins=20):
    props = measure.regionprops(measure.label(mask))
    if not props:
        return [float("nan")]*n_bins
    p = max(props, key=lambda r: r.area)
    cy, cx = p.centroid
    yy, xx = np.indices(image.shape)
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    rmax = rr[mask].max()
    prof = []
    for i in range(n_bins):
        r0 = i*rmax/n_bins
        r1 = (i+1)*rmax/n_bins
        sel = (rr>=r0)&(rr<r1)&mask
        prof.append(float(image[sel].mean()) if sel.sum()>0 else float("nan"))
    return prof

def texture_index(img, mask):
    vals = img[mask]
    if vals.size<10:
        return float("nan")
    var = float(np.var(vals))
    mean = float(np.mean(vals)) if np.mean(vals)>0 else np.nan
    return float(var/mean) if (mean and np.isfinite(mean)) else float("nan")

ROLE_PATTERNS = {
    "dapi": re.compile(r"(dapi|hoechst|nuc(lei|clear)?)", re.I),
    "immune": re.compile(r"(cd3|cd8|immune|t[-_ ]?cell|tcell)", re.I),
    "tumor": re.compile(r"(phase|bf|bright|tumor|ck|actin)", re.I),
}

def is_junk_mac(path):
    name = os.path.basename(path)
    return ("__MACOSX" in path) or name.startswith("._") or name.startswith(".")

def guess_role_from_name(name):
    for role, pat in ROLE_PATTERNS.items():
        if pat.search(name):
            return role
    return "unknown"

def canonical_stem(name):
    tokens = ["dapi","hoechst","nuclei","nuclear","nuc","cd3","cd8","immune","tcell","t-cell","phase","bf","bright","tumor","ck","actin"]
    s = name.lower()
    for t in tokens:
        s = re.sub(rf"([_\-\s])?{t}([_\-\s])?", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or name

def group_multichannel(files):
    groups = {}
    for p in files:
        if is_junk_mac(p):
            continue
        base = os.path.splitext(os.path.basename(p))[0]
        role = guess_role_from_name(base)
        stem = canonical_stem(base)
        g = groups.setdefault(stem, {"all": []})
        g["all"].append(p)
        if role != "unknown":
            g[role] = p
    return groups

def safe_imread(path):
    if path is None:
        raise ValueError("No path provided")
    if not isinstance(path, str):
        path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # reject mac resource files early
    if is_junk_mac(path):
        raise IOError(f"Junk macOS resource file: {path}")
    img = io.imread(path)
    if img is None:
        raise IOError(f"Failed to read: {path}")
    return img

def overlay_png(base_img, mask, rim, dpi=200):
    import matplotlib.pyplot as plt
    from io import BytesIO
    plt.figure(figsize=(5,5))
    plt.imshow(exposure.equalize_hist(base_img), cmap="gray")
    b = segmentation.find_boundaries(mask, mode="outer")
    plt.contour(b, levels=[0.5])
    plt.contour(rim, levels=[0.5], linestyles="dashed")
    plt.axis("off")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()

def process_entry(entry, px_per_micron, rim_width_microns, prefer_channel="tumor"):
    base_path = entry.get(prefer_channel) or entry.get("tumor") or (entry.get("all")[0] if entry.get("all") else None)
    if base_path is None:
        raise ValueError("No valid image found for this group (missing base channel).")
    img = safe_imread(base_path)
    base_img = _to_gray(img)
    base_img_bs = background_subtract(base_img, radius_px=int(50/px_per_micron))

    mask = autoseg_from_phase(base_img_bs)
    rim, core = rim_core(mask, rim_px=int(rim_width_microns*px_per_micron))

    px_per_mm = px_per_micron*1000.0
    area_mm2 = mask.sum()/(px_per_mm**2) if mask.any() else np.nan

    boundary_mean_intensity = float(base_img_bs[rim].mean()) if rim.any() else np.nan
    core_mean_intensity = float(base_img_bs[core].mean()) if core.any() else np.nan
    infiltration_index_phase = float(core_mean_intensity/boundary_mean_intensity) if (np.isfinite(core_mean_intensity) and np.isfinite(boundary_mean_intensity) and boundary_mean_intensity>0) else np.nan
    tex_idx = texture_index(base_img_bs, mask)
    radial_phase = radial_profile(base_img_bs, mask, n_bins=50)

    immune_boundary = immune_core = immune_infiltration = np.nan
    radial_immune = None
    if "immune" in entry and not is_junk_mac(entry["immune"]):
        try:
            im = safe_imread(entry["immune"])
            im = _to_gray(im)
            im_bs = background_subtract(im, radius_px=int(50/px_per_micron))
            immune_boundary = float(im_bs[rim].mean()) if rim.any() else np.nan
            immune_core = float(im_bs[core].mean()) if core.any() else np.nan
            immune_infiltration = float(immune_core/immune_boundary) if (np.isfinite(immune_core) and np.isfinite(immune_boundary) and immune_boundary>0) else np.nan
            radial_immune = radial_profile(im_bs, mask, n_bins=50)
        except Exception:
            pass

    nuclei_count = nuclei_density = np.nan
    if "dapi" in entry and not is_junk_mac(entry["dapi"]):
        try:
            d = safe_imread(entry["dapi"])
            d = _to_gray(d)
            d_bs = background_subtract(d, radius_px=int(50/px_per_micron))
            th = filters.threshold_otsu(d_bs[mask]) if mask.any() else filters.threshold_otsu(d_bs)
            bw = (d_bs > th) & mask
            lab = measure.label(bw)
            nuclei_count = int(lab.max())
            area = mask.sum()
            nuclei_density = float(nuclei_count/area) if area>0 else np.nan
        except Exception:
            pass

    ov_png = overlay_png(base_img_bs, mask, rim)
    row = {
        "id": os.path.basename(base_path),
        "has_dapi": int("dapi" in entry),
        "has_immune": int("immune" in entry),
        "area_mm2": float(area_mm2) if np.isfinite(area_mm2) else np.nan,
        "phase_boundary_mean": boundary_mean_intensity,
        "phase_core_mean": core_mean_intensity,
        "phase_infiltration_index": infiltration_index_phase,
        "phase_texture_index": tex_idx,
        "immune_boundary_mean": immune_boundary,
        "immune_core_mean": immune_core,
        "immune_infiltration_index": immune_infiltration,
        "nuclei_count_proxy": nuclei_count,
        "nuclei_density_proxy": nuclei_density,
        "radial_phase": radial_phase,
        "radial_immune": radial_immune
    }
    return row, ov_png
