
import io
import os
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import detrend
from skimage import io as skio, util, measure, morphology

# ---------------------------
# ---------- Utils ----------
# ---------------------------

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

def to_gray(arr):
    """Convert image to float grayscale [0,1]."""
    arr = util.img_as_float(arr)
    if arr.ndim == 3:
        # average across channels (çok kanallı ise basit ortalama)
        arr = arr.mean(axis=-1)
    return arr

def radial_profile(img, mask, n_bins=64):
    """Mean intensity from center to rim within mask in n_bins."""
    lab = measure.label(mask.astype(np.uint8))
    props = measure.regionprops(lab)
    if not props:
        return None
    p = max(props, key=lambda r: r.area)
    cy, cx = p.centroid
    yy, xx = np.indices(img.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr[mask].max()
    prof = []
    for i in range(n_bins):
        r0 = i * rmax / n_bins
        r1 = (i + 1) * rmax / n_bins
        sel = (rr >= r0) & (rr < r1) & mask
        prof.append(float(img[sel].mean()) if sel.sum() > 0 else np.nan)
    prof = np.array(prof, float)
    # Fill NaNs by linear interpolation if possible
    if np.isnan(prof).any():
        idx = np.where(~np.isnan(prof))[0]
        if idx.size >= 2:
            prof = np.interp(np.arange(len(prof)), idx, prof[idx])
        else:
            return None
    return prof

def rosenstein_lle(ts, m=3, tau=1, min_sep=8, max_iter=20):
    """Lyapunov exponent via Rosenstein's method on 1D series."""
    ts = np.asarray(ts, float)
    ts = detrend(ts)
    std = ts.std()
    if std > 0:
        ts = (ts - ts.mean()) / std
    N = len(ts) - (m - 1) * tau
    if N <= m + 1:
        return np.nan, 0.0, 0
    # Phase space embedding
    Y = np.zeros((N, m))
    for i in range(m):
        Y[:, i] = ts[i * tau: i * tau + N]
    # Nearest neighbors with temporal separation
    nn = np.full(N, -1, int)
    for i in range(N):
        idx = np.arange(N)
        idx = idx[np.abs(idx - i) > min_sep]
        if idx.size == 0:
            continue
        d = np.linalg.norm(Y[idx] - Y[i], axis=1)
        nn[i] = idx[np.argmin(d)]
    # Divergence curves
    curves = []
    for i in range(N):
        j = nn[i]
        if j < 0:
            continue
        L = min(max_iter, N - max(i, j) - 1)
        if L <= 3:
            continue
        dist = []
        for k in range(1, L):
            dist.append(np.linalg.norm(Y[i + k] - Y[j + k]))
        dist = np.array(dist)
        dist = dist[dist > 0]
        if dist.size:
            curves.append(np.log(dist))
    if not curves:
        return np.nan, 0.0, 0
    Lmin = min(map(len, curves))
    M = np.vstack([c[:Lmin] for c in curves])
    y = M.mean(axis=0)
    x = np.arange(1, Lmin + 1)
    A = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = beta[0]
    yhat = A @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r2), int(len(curves))

def fractal_dimension(mask, k_min=2, steps=6):
    """
    Box-counting fractal dimension on mask boundary.
    Returns (FD, R2). Uses log(1/boxsize) vs log(count) linear fit.
    """
    er = morphology.binary_erosion(mask)
    boundary = mask ^ er
    Z = boundary.astype(np.uint8)
    n = min(Z.shape)
    k_max = n // 4 if n >= 16 else max(2, n // 2)
    sizes = np.unique(np.logspace(np.log10(k_min), np.log10(k_max), num=steps, dtype=int))
    sizes = sizes[sizes >= 2]
    if len(sizes) < 3:
        return np.nan, 0.0
    counts = []
    for k in sizes:
        pad0 = (k - (Z.shape[0] % k)) % k
        pad1 = (k - (Z.shape[1] % k)) % k
        Zp = np.pad(Z, ((0, pad0), (0, pad1)), mode='constant', constant_values=0)
        S = np.add.reduceat(
            np.add.reduceat(Zp, np.arange(0, Zp.shape[0], k), axis=0),
            np.arange(0, Zp.shape[1], k), axis=1
        )
        counts.append((S > 0).sum())
    counts = np.array(counts, float)
    m = counts > 0
    sizes, counts = sizes[m], counts[m]
    if len(sizes) < 3:
        return np.nan, 0.0
    x = np.log(1.0 / sizes)
    y = np.log(counts)
    A = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    dim = beta[0]
    yhat = A @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(dim), float(r2)

def short_label(name, keep=24):
    base = os.path.basename(str(name))
    base = base.replace(".tif","").replace(".png","").replace(".jpg","")
    return base[-keep:] if len(base) > keep else base

def _is_mask_name(p):
    low = p.lower()
    base = os.path.basename(low)
    return ("mask" in low) or base.endswith("_mask.tif") or base.endswith("_mask.png") or base.endswith("_mask.jpg") or base.endswith("_mask.jpeg")

def bytes_to_bool_mask(b):
    arr = skio.imread(io.BytesIO(b))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)

def parse_zip_images_and_masks(zf_bytes):
    """
    ZIP içinden (görüntüler, maskeler) çıkarır.
    Döndürür:
      images_list: [(name, bytes), ...]
      masks_dict:  {token: bool_mask_numpy, ...}
      single_mask_bool: tek maske tespit edilirse numpy (0/1), yoksa None
    """
    images_list = []
    mask_files = []  # [(name, bytes)]
    with zipfile.ZipFile(io.BytesIO(zf_bytes), "r") as zf:
        for name in zf.namelist():
            low = name.lower()
            if low.endswith("/") or "__macosx" in low or os.path.basename(low).startswith("._"):
                continue
            if not low.endswith(IMG_EXTS):
                continue
            data = zf.read(name)
            if _is_mask_name(name):
                mask_files.append((name, data))
            else:
                images_list.append((name, data))

    masks_dict = {}
    for (nm, b) in mask_files:
        base = os.path.basename(nm)
        base_noext = os.path.splitext(base)[0]
        token = base_noext.replace("_mask", "")
        masks_dict[token] = bytes_to_bool_mask(b)

    single_mask_bool = None
    if len(masks_dict) == 1:
        single_mask_bool = list(masks_dict.values())[0]

    return images_list, masks_dict, single_mask_bool

def match_mask_for_image(img_name, mask_dict, default_mask=None):
    """
    1) default_mask verilmişse onu kullan.
    2) Yoksa mask_dict içinde anahtar (token) img_name içinde geçiyorsa onu kullan.
    """
    if default_mask is not None:
        return default_mask
    for token, msk in mask_dict.items():
        if token in img_name:
            return msk
    return None

# ---------------------------
# --------- UI/UX -----------
# ---------------------------

st.set_page_config(page_title="3D Immune Chaos — FD & LLE", layout="wide")
st.title("3D Immune Chaos — Organoid/Chip Görüntülerinde FD + LLE Analizi")

st.markdown(
    """
Bu arayüz, **maskeye dayalı ROI** ile **Fraktal Boyut (FD)** ve **Lyapunov Üssü (LLE)** hesaplar:
- **FD**: maskenin *sınır* geometrisinin karmaşıklığı (box-counting).
- **LLE**: mask içi **merkez→rim** radyal yoğunluk profilinden Rosenstein yöntemiyle kaotik duyarlılık.

**Kullanım:**
1. Solda **görüntü(ler)**ini yükle (`.tif/.png/.jpg`).
2. **Maske** yükle (tek maske tüm görüntülere uygulanır) **veya** “Gelişmiş Eşleştirme” ile **çoklu maske** + token eşleştir.
3. **ZIP** yükleme desteğiyle, görüntü+maskeyi tek dosyada da gönderebilirsin.
4. “Analizi Çalıştır” de; tablo, grafik ve **CSV indir** butonu oluşur.
    """
)

with st.sidebar:
    st.header("Veri Yükleme")
    imgs = st.file_uploader("Görüntüler (birden fazla seçebilirsin)", type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True)
    single_mask = st.file_uploader("Tek Maske (opsiyonel — tüm görüntülere uygulanır)", type=["tif","tiff","png","jpg","jpeg"])

    st.markdown("---")
    st.subheader("📦 ZIP ile Yükleme (opsiyonel)")
    zip_file = st.file_uploader("Görüntü + Maske içeren .zip yükle", type=["zip"])
    use_zip = st.checkbox("ZIP içeriğini kullan", value=False)

    st.markdown("---")
    st.subheader("Gelişmiş Eşleştirme (opsiyonel)")
    st.caption("Birden çok maskeyi, **anahtar token** ile eşleştirebilirsin. Örn: token=`Human_1_004h`")
    use_multi = st.checkbox("Çoklu maske-token eşleştirme kullan", value=False)
    mask_tokens = {}
    if use_multi:
        n_masks = st.number_input("Kaç farklı maske-token gireceksin?", min_value=1, max_value=10, value=1, step=1)
        for i in range(n_masks):
            with st.expander(f"Maske #{i+1}"):
                token = st.text_input(f"Token #{i+1} (örn. Human_1_004h)", key=f"token_{i}")
                mfile = st.file_uploader(f"Maske dosyası #{i+1}", type=["tif","tiff","png","jpg","jpeg"], key=f"mask_{i}")
                if token and mfile:
                    arr = skio.imread(mfile)
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    mask_tokens[token] = (arr > 0).astype(np.uint8)

    run = st.button("Analizi Çalıştır")

# ---------------------------
# ------- Processing --------
# ---------------------------

if run:
    # ZIP kullanımını hazırla
    zip_images = []
    zip_masks_dict = {}
    zip_single_mask = None

    if use_zip:
        if zip_file is None:
            st.error("ZIP kullanımını seçtiniz ama dosya yüklemediniz.")
            st.stop()
        try:
            zip_images, zip_masks_dict, zip_single_mask = parse_zip_images_and_masks(zip_file.read())
            if len(zip_images) == 0:
                st.error("ZIP içinde görüntü bulunamadı. .tif/.png/.jpg dosyaları ekleyin.")
                st.stop()
        except Exception as e:
            st.error(f"ZIP okunamadı: {e}")
            st.stop()

    # Giriş doğrulama
    if not use_zip and not imgs:
        st.error("Lütfen en az bir görüntü yükleyin (veya ZIP kullanın).")
        st.stop()

    if not use_zip and (single_mask is None and not mask_tokens):
        st.warning("Maske yüklemediniz. Analiz için maske gereklidir (veya ZIP içinde maske sunun).")
        st.stop()

    # Maske hazırlığı
    single_mask_bool = None
    mask_dict_bool = {}

    if use_zip:
        single_mask_bool = zip_single_mask
        mask_dict_bool = zip_masks_dict
        # ZIP'te maske yoksa, sidebar’dakileri yedek olarak al
        if single_mask_bool is None and not mask_dict_bool:
            if single_mask is not None:
                arr = skio.imread(single_mask)
                if arr.ndim == 3: arr = arr[..., 0]
                single_mask_bool = (arr > 0).astype(np.uint8)
            if mask_tokens and not mask_dict_bool:
                mask_dict_bool = mask_tokens
    else:
        if single_mask is not None:
            arr = skio.imread(single_mask)
            if arr.ndim == 3: arr = arr[..., 0]
            single_mask_bool = (arr > 0).astype(np.uint8)
        if mask_tokens:
            mask_dict_bool = mask_tokens

    # İşlenecek görüntü listesi
    rows = []
    profiles = []
    progress = st.progress(0.0)

    if use_zip:
        total = len(zip_images)
        iterable = enumerate(zip_images)
    else:
        total = len(imgs)
        iterable = enumerate(imgs)

    for i, item in iterable:
        try:
            if use_zip:
                img_name, img_bytes = item
                img_arr = skio.imread(io.BytesIO(img_bytes))
            else:
                img_name, img_arr = item.name, skio.imread(item)

            img = to_gray(img_arr)

            # Maske eşle
            matched_mask = None
            if single_mask_bool is not None or mask_dict_bool:
                matched_mask = match_mask_for_image(img_name, mask_dict_bool, default_mask=single_mask_bool)

            if matched_mask is None:
                rows.append({
                    "image": img_name, "mask_source": "none",
                    "LLE": np.nan, "LLE_R2": 0.0, "LLE_pairs": 0,
                    "FD": np.nan, "FD_R2": 0.0, "note": "no_matching_mask"
                })
                progress.progress((i + 1) / max(1, total))
                continue

            # Radyal profil + LLE
            prof = radial_profile(img, matched_mask, n_bins=64)
            if prof is None or len(prof) < 8:
                LLE, LLE_R2, LLE_pairs = np.nan, 0.0, 0
                note = "no_profile"
            else:
                LLE, LLE_R2, LLE_pairs = rosenstein_lle(prof, m=3, tau=1, min_sep=8, max_iter=20)
                note = "ok"
                profiles.append(pd.DataFrame({
                    "image": img_name,
                    "bin": np.arange(len(prof)),
                    "radial_mean": prof
                }))

            # FD (mask boundary)
            FD, FD_R2 = fractal_dimension(matched_mask, k_min=2, steps=6)

            rows.append({
                "image": img_name,
                "mask_source": ("zip_single_mask" if use_zip and zip_single_mask is not None
                                else "zip_token_match" if use_zip else
                                "single_mask" if single_mask_bool is not None else "token_match"),
                "LLE": LLE, "LLE_R2": LLE_R2, "LLE_pairs": LLE_pairs,
                "FD": FD, "FD_R2": FD_R2,
                "note": note
            })

        except Exception as ex:
            rows.append({
                "image": img_name if use_zip else item.name,
                "mask_source": "error",
                "LLE": np.nan, "LLE_R2": 0.0, "LLE_pairs": 0,
                "FD": np.nan, "FD_R2": 0.0, "note": f"error: {ex}"
            })

        progress.progress((i + 1) / max(1, total))

    df = pd.DataFrame(rows)
    st.success("Analiz tamamlandı.")

    # ---------------------------
    # ----- Results + Plots -----
    # ---------------------------

    st.subheader("Sonuç Tablosu")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("CSV indir", data=csv_bytes, file_name="fd_lle_results.csv", mime="text/csv")

    # Scatter FD vs LLE
    valid = df.dropna(subset=["FD", "LLE"])
    if len(valid) > 0:
        st.subheader("FD vs LLE (Saçılım)")
        fig = plt.figure()
        plt.scatter(valid["FD"].values, valid["LLE"].values)
        for _, row in valid.iterrows():
            plt.annotate(short_label(row["image"]), (row["FD"], row["LLE"]),
                         fontsize=7, xytext=(2, 2), textcoords='offset points')
        plt.xlabel("Fraktal Boyut (boundary, box-counting)")
        plt.ylabel("Lyapunov Üssü (Rosenstein)")
        plt.title("FD vs LLE")
        st.pyplot(fig)

    # Bar: LLE
    st.subheader("Görüntü Başına LLE")
    fig = plt.figure()
    plt.bar(range(len(df)), df["LLE"].values)
    plt.xticks(range(len(df)), [short_label(s) for s in df["image"].values], rotation=90, fontsize=7)
    plt.ylabel("LLE")
    plt.title("LLE per image")
    st.pyplot(fig)

    # Bar: FD
    st.subheader("Görüntü Başına FD")
    fig = plt.figure()
    plt.bar(range(len(df)), df["FD"].values)
    plt.xticks(range(len(df)), [short_label(s) for s in df["image"].values], rotation=90, fontsize=7)
    plt.ylabel("FD (boundary)")
    plt.title("FD per image")
    st.pyplot(fig)

    # Radial profiles (optional)
    if len(profiles) > 0:
        st.subheader("Radyal Profiller (Merkez → Rim)")
        prof_df = pd.concat(profiles, ignore_index=True)
        sel_img = st.selectbox("Profilini gör (1 görüntü seç):", sorted(prof_df["image"].unique()))
        sub = prof_df[prof_df["image"] == sel_img].sort_values("bin")
        fig = plt.figure()
        plt.plot(sub["bin"].values, sub["radial_mean"].values, marker="o")
        plt.xlabel("Radial bin (0=merkez → 63=rim)")
        plt.ylabel("Mean intensity")
        plt.title(f"Radial profile — {short_label(sel_img)}")
        st.pyplot(fig)

        # Profilleri CSV indir
        prof_csv = prof_df.to_csv(index=False).encode("utf-8")
        st.download_button("Radyal Profiller CSV indir", data=prof_csv, file_name="radial_profiles.csv", mime="text/csv")

else:
    st.info("Soldan görüntü(ler) ve maske(ler) seç veya ZIP yükle, ardından **Analizi Çalıştır**.")
