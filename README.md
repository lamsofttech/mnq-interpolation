---
title: MNQ Interpolation Methods
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Four interpolation methods on intraday MNQ prices
---

# Comparing Interpolation Methods for Reconstructing Intraday MNQ Futures Prices

Lameck Nyakweba · Mathematics 5670-08, Fall 2026 · Austin Peay State University

This project compares linear interpolation, natural cubic splines, PCHIP (piecewise cubic Hermite), and local Newton divided-difference interpolation on 1-minute Micro E-mini Nasdaq-100 futures (MNQ) prices. For each trading day, prices are kept every 5, 15, or 30 minutes, the prices in between are hidden, and each method rebuilds them from the kept prices only. The estimates are then compared with the true prices.

## Files

| File | Purpose |
|---|---|
| `interpolation_project.ipynb` | The full analysis with explanations, tables, and figures |
| `interpolation.py` | The methods and the experiment, shared by the app and the web page |
| `figures.py` | The charts, shared by the app and the web page |
| `index.html`, `web.py` | The Hugging Face page: runs the Python code in the browser with Pyodide |
| `app.py` | A Gradio version of the app for running on your own computer |
| `download_mnq.py` | Downloads MNQ 1-minute bars from Yahoo Finance into `data/` |
| `data/mnq_1m_raw.csv` | The saved data snapshot |
| `data/download_log.json` | When and how the data were downloaded |

## Run locally

```
pip install -r requirements.txt gradio
python download_mnq.py      # optional: refresh the data snapshot
python app.py               # opens the app at http://127.0.0.1:7860
```

Or open `interpolation_project.ipynb` in VS Code or Jupyter and choose Run All.

## Deployment

The Hugging Face Space is a free static Space that serves `index.html`. The page loads Pyodide (Python compiled for the browser), copies in `interpolation.py`, `figures.py`, `web.py`, and the data snapshot, and runs the full experiment in the visitor's browser in under a minute.

Every push to the `main` branch on GitHub runs `.github/workflows/sync-to-hugging-face.yml`, which copies the repository to the Space.

Market data come from Yahoo Finance and are used for educational purposes only. Nothing here is trading or investment advice.
