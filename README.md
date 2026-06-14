# Topological Data Analysis with Macro‑Parameterised Filtrations

Applies persistent homology to ETF correlation distance matrices. The Rips filtration distance is scaled by a composite macro factor (from VIX, DXY, yields), adapting the topological analysis to market conditions. The per‑ETF score is the node degree in the macro‑adjusted Rips graph – a measure of topological centrality.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Distance matrix = 1 - |correlation|
- Composite macro factor via PCA on all macro variables
- Macro‑scaled filtration distance = base * (1 - macro_factor * 0.5)
- Score = node degree (number of close neighbours at macro‑scaled distance)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-tda-macro-filtration-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF is topologically central under current macro conditions.
- Low score → ETF is peripheral.

## Requirements

See `requirements.txt`.
