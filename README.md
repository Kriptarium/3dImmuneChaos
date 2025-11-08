# 3D Immune Chaos Analyzer

**Purpose:**  
This Streamlit app computes **Fractal Dimension (FD)** and **Lyapunov Exponent (LLE)** from organoid or chip microscopy images using mask-based regions of interest.

---

## 🚀 Usage

1. Open the app (either locally or on Streamlit Cloud).
2. Upload microscopy images (`.tif`, `.png`, `.jpg`).
3. Upload a corresponding binary mask:
   - **Single mask:** applied to all images.
   - **Multiple masks:** use the “Gelişmiş Eşleştirme” section to match by token (e.g. `Human_1_004h`).
4. Click **Analizi Çalıştır**.
5. View results:
   - Summary table (LLE, FD per image)
   - Scatter and bar plots
   - Downloadable CSV files.

---

## 🧮 Scientific Rationale

- **FD (Fractal Dimension):** quantifies morphological boundary complexity.  
  → Higher FD = more irregular or invasive boundary.

- **LLE (Lyapunov Exponent):** quantifies chaotic sensitivity of the radial intensity profile (center → rim).  
  → Higher LLE = stronger internal heterogeneity or immune infiltration.

---

## 🧰 Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

- Create a new app, connect your GitHub repo.
- Set **Main file path:** `app.py`
- Click **Deploy**.
