
import streamlit as st
import os, zipfile, tempfile, json
import numpy as np
import pandas as pd

from preprocess_core import group_multichannel, process_entry
from chaos_metrics import lyapunov_rosenstein, higuchi_fd
from report_utils import save_simple_pdf

st.set_page_config(page_title="3dImmuneChaos", layout="wide")
st.title("3dImmuneChaos — Çok Kanallı Ön İşleme + Kaotik Metrikler (Patched)")

with st.sidebar:
    st.header("Genel Ayarlar")
    px_per_micron = st.number_input("Pixels per micron (px/µm)", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
    rim_width = st.number_input("Rim genişliği (µm)", min_value=2.0, max_value=200.0, value=20.0, step=2.0)
    prefer_channel = st.selectbox("Segmentasyon kanalı", ["tumor", "immune", "dapi"], index=0)

tabs = st.tabs(["1) Veri & Ön İşleme", "2) Kaotik Metrikler + Rapor", "3) Günlük (Debug)"])

if "rows" not in st.session_state:
    st.session_state.rows = []
if "overlays" not in st.session_state:
    st.session_state.overlays = []
if "debug" not in st.session_state:
    st.session_state.debug = []

with tabs[0]:
    st.subheader("Veri yükle")
    mode = st.radio("Yükleme yöntemi", ["ZIP (önerilir)", "Tekil dosyalar"])
    work_dir = tempfile.mkdtemp()
    all_files = []

    def _collect_images(base):
        cnt = 0
        for rootd, dirs, files in os.walk(base):
            for name in files:
                if name.lower().endswith((".png",".tif",".tiff",".jpg",".jpeg")):
                    all_files.append(os.path.join(rootd, name))
                    cnt += 1
        return cnt

    if mode == "ZIP (önerilir)":
        up = st.file_uploader("ZIP yükle (dosya adlarında dapi / immune / phase / tumor vs. geçmeli)", type=["zip"])
        if up is not None:
            zpath = os.path.join(work_dir, "upload.zip")
            with open(zpath, "wb") as f:
                f.write(up.getvalue())
            try:
                with zipfile.ZipFile(zpath, "r") as z:
                    z.extractall(work_dir)
                cnt = _collect_images(work_dir)
                st.info(f"{cnt} görüntü algılandı.")
            except Exception as e:
                st.error(f"ZIP açılamadı: {e}")
    else:
        imgs = st.file_uploader("Görüntü dosyaları yükle", type=["png","tif","tiff","jpg","jpeg"], accept_multiple_files=True)
        if imgs:
            for up in imgs:
                p = os.path.join(work_dir, up.name)
                with open(p, "wb") as f:
                    f.write(up.getvalue())
            cnt = _collect_images(work_dir)
            st.info(f"{cnt} görüntü yüklendi.")

    if st.button("Ön işle ve metrikleri çıkar"):
        st.session_state.debug.clear()
        if not all_files:
            st.error("Önce veri yükleyin.")
        else:
            groups = group_multichannel(all_files)
            rows = []
            overlays = []
            skipped = 0
            for stem, entry in groups.items():
                try:
                    row, ov = process_entry(entry, px_per_micron, rim_width, prefer_channel=prefer_channel)
                    rows.append(row)
                    overlays.append((row["id"], ov))
                except Exception as ex:
                    skipped += 1
                    st.session_state.debug.append(f"[SKIP] stem={stem} error={ex}")
            if not rows:
                st.error("Hiç örnek işlenemedi. 'Günlük (Debug)' sekmesine bakın.")
            else:
                df = pd.DataFrame(rows)
                st.session_state.rows = rows
                st.session_state.overlays = overlays
                st.success(f"{len(rows)} örnek işlendi. (Atlanan: {skipped})")
                st.dataframe(df, use_container_width=True)
                st.download_button("metrics.csv indir", data=df.to_csv(index=False).encode("utf-8"),
                                   file_name="metrics.csv", mime="text/csv")
                if overlays:
                    st.subheader("QA Overlays")
                    cols = st.columns(3)
                    for i,(name, png) in enumerate(overlays):
                        with cols[i%3]:
                            st.image(png, caption=name, use_container_width=True)

with tabs[1]:
    st.subheader("Kaotik metrikler")
    if not st.session_state.rows:
        st.info("Önce 'Veri & Ön İşleme' sekmesinde işlem yapın.")
    else:
        df = pd.DataFrame(st.session_state.rows)
        source = st.selectbox("Radyal profil kaynağı", ["radial_immune (varsa)", "radial_phase"], index=0)

        series_map = {}
        for r in st.session_state.rows:
            sid = r["id"]
            if source.startswith("radial_immune") and r.get("radial_immune") is not None:
                series_map[sid] = r["radial_immune"]
            else:
                series_map[sid] = r["radial_phase"]

        c1, c2, c3 = st.columns(3)
        with c1:
            m = st.number_input("Gömme m", 2, 10, 3)
        with c2:
            tau = st.number_input("Gecikme tau", 1, 10, 1)
        with c3:
            min_sep = st.number_input("Min ayrım", 1, 50, 5)
        max_iter = st.slider("Max iter", 5, 100, 20)

        lyap_rows = []
        higuchi_rows = []
        for sid, ts in series_map.items():
            ts = np.array([t for t in ts if t is not None and np.isfinite(t)])
            if len(ts) < 8:
                lle, r2, pairs = np.nan, 0.0, 0
                hfd = np.nan
            else:
                lle, r2, pairs = lyapunov_rosenstein(ts, m=m, tau=tau, min_separation=min_sep, max_iter=max_iter)
                hfd = higuchi_fd(ts, kmax=min(10, max(6, len(ts)//3)))
            lyap_rows.append({"id": sid, "LLE": lle, "R2": r2, "pairs": pairs})
            higuchi_rows.append({"id": sid, "HiguchiFD": hfd})

        df_lyap = pd.DataFrame(lyap_rows).set_index("id")
        df_hfd = pd.DataFrame(higuchi_rows).set_index("id")
        df_merged = df.set_index("id").join(df_lyap).join(df_hfd).reset_index()
        st.dataframe(df_merged, use_container_width=True)

        st.download_button("chaos_metrics.csv indir",
                           data=df_merged.to_csv(index=False).encode("utf-8"),
                           file_name="chaos_metrics.csv",
                           mime="text/csv")

        st.subheader("Rapor (PDF)")
        title = st.text_input("Rapor başlığı", "3dImmuneChaos — Kaotik Metrik Raporu")
        if st.button("PDF oluştur"):
            pairs = []
            def safe_mean(col):
                import numpy as np
                try:
                    return f"{np.nanmean(col.astype(float)):.4f}"
                except Exception:
                    return "NaN"
            pairs.append(("Örnek sayısı", len(df_merged)))
            for col in ["phase_infiltration_index","phase_texture_index","immune_infiltration_index","LLE","HiguchiFD"]:
                if col in df_merged.columns:
                    try:
                        pairs.append((f"Ort {col}", safe_mean(df_merged[col].values)))
                    except Exception:
                        pass
            pdf_path = os.path.join(tempfile.gettempdir(), "3dImmuneChaos_report.pdf")
            save_simple_pdf(pdf_path, title, pairs)
            with open(pdf_path, "rb") as f:
                st.download_button("PDF indir", data=f.read(), file_name="3dImmuneChaos_report.pdf", mime="application/pdf")

with tabs[2]:
    st.subheader("Günlük (Debug)")
    if st.session_state.debug:
        for line in st.session_state.debug:
            st.code(line)
    else:
        st.write("Henüz günlük kaydı yok.")
