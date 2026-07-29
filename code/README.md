# Fe550 TMT — Mechanical Property Predictive Model

Code for correlating process/chemistry parameters with tensile properties of
Fe550 25 mm Merchant-Mill TMT bars and predicting YS, UTS, % Elongation and the
UTS/YS ratio before physical testing.

## Setup (VS Code)

```bash
cd code
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate
pip install -r requirements.txt
```

The scripts read `../Data for Corelation.xlsx` automatically (path set in
`config.py`), so keep the `code/` folder inside the project folder.

## Run order

```bash
python data_prep.py            # Stage 1 — clean + summary  -> outputs/cleaned_data.csv
python correlation_analysis.py # Stage 2 — correlations     -> outputs/correlation_*.png/.csv
python train_models.py         # Stage 3-6 — train/test/eval -> models/*.joblib, outputs/metrics_summary.json
python predict.py              # demo inference on one row
streamlit run app.py           # interactive web app to enter inputs & predict
```

## Streamlit app

`app.py` is a browser UI: enter chemistry/rolling/cooling parameters in the
sidebar, click **Predict properties**, and it shows predicted YS, UTS,
% Elongation, UTS/YS and a green/red **acceptable / not-acceptable** verdict
against the BHEL/L&T window. It requires the models from `train_models.py`;
if they're missing it tells you to run that first. Start it with
`streamlit run app.py` from inside the `code/` folder.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Paths, column names, acceptance thresholds |
| `data_prep.py` | Load + clean the workbook (Stage 1) |
| `correlation_analysis.py` | Pearson/Spearman correlations + heatmap (Stage 2) |
| `train_models.py` | Feature selection, model comparison, training, hold-out test, evaluation, feature importance, acceptance prediction (Stages 3-6) |
| `predict.py` | Load saved models and predict properties for new inputs |
| `app.py` | Streamlit web app: enter parameters, get predictions + acceptance verdict |

## Model choice

Individual parameters correlate only weakly and linearly with the properties
(all |Pearson r| ≤ ~0.2), but the metallurgy is governed by **non-linear
interactions**. `train_models.py` therefore compares a **Linear Regression**
baseline, a **Random Forest** and **Gradient Boosting** by 5-fold cross-validated
R², and selects the best model per property. Random Forest is expected to win:
it captures interactions automatically, needs no scaling, resists outliers,
works well on a few-hundred-row dataset, and exposes feature importance for
process engineers. The header comment in `train_models.py` explains this in full.

## Acceptance rule (BHEL / L&T)

A bar is "acceptable" when **YS ≤ 660 MPa** and **1.15 ≤ UTS/YS ≤ 1.25**.
`train_models.py` reports how well the model reproduces the acceptance decision
on the hold-out set; `predict.py` flags acceptability for any new input.
