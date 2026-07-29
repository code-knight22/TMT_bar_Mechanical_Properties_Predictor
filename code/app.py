"""
Streamlit app — TMT bar mechanical-property predictor.

Loads the models saved by train_models.py and predicts YS, UTS, % Elongation
and the UTS/YS ratio from operator-entered process/chemistry parameters, then
flags whether the bar meets the BHEL/L&T acceptance window.

Run:
    cd code
    streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import config as C

# ---------------------------------------------------------------- page
st.set_page_config(page_title="TMT bar Property Predictor",
                   page_icon="🛠️", layout="wide")

# Input widget spec:  key -> (label, min, max, default, step)
INPUT_SPEC = {
    "C":          ("%C",                       0.15, 0.30, 0.20, 0.005),
    "Mn":         ("%Mn",                      0.40, 1.00, 0.71, 0.01),
    "RPM":        ("12th-stand RPM",           400.0, 600.0, 500.0, 1.0),
    "LinSpeed":   ("Linear speed (m/s)",       7.0, 10.5, 8.45, 0.01),
    "T_11std":    ("11th-stand temp (°C)",     980.0, 1150.0, 1052.0, 1.0),
    "WaterFlow":  ("Water flow (Nm³/hr)",      500.0, 850.0, 646.0, 1.0),
    "WaterPress": ("Water pressure (kg/cm²)",  8.0, 18.0, 11.7, 0.1),
    "T_7pipe":    ("Temp after 7th pipe (°C)", 20.0, 260.0, 83.0, 1.0),
    "EqualTemp":  ("Equalisation temp (°C)",   460.0, 590.0, 520.0, 1.0),
}


@st.cache_resource
def load_models():
    models = {}
    for target in C.OUTPUTS:
        path = C.MODEL_DIR / f"model_{target}.joblib"
        if not path.exists():
            return None
        models[target] = joblib.load(path)
    return models


# ---------------------------------------------------------------- header
st.title("🛠️ TMT bar Mechanical-Property Predictor")
st.caption("25 mm Merchant-Mill bars · IS 1786 Fe 550D · predicts YS, UTS, "
           "% Elongation and UTS/YS before physical testing")

models = load_models()
if models is None:
    st.error(
        "No trained models found in the `models/` folder.\n\n"
        "Run **`python train_models.py`** first to create "
        "`model_YS.joblib`, `model_UTS.joblib`, `model_Elong.joblib` and "
        "`model_Ratio.joblib`, then reload this page.")
    st.stop()

features = models["YS"]["features"]

# ---------------------------------------------------------------- inputs
st.sidebar.header("Input parameters")
st.sidebar.caption("Set the rolling / cooling / chemistry values, then predict.")

params = {}
groups = {
    "Chemistry": ["C", "Mn"],
    "Rolling": ["RPM", "LinSpeed", "T_11std"],
    "Quenching & cooling": ["WaterFlow", "WaterPress", "T_7pipe", "EqualTemp"],
}
for group, keys in groups.items():
    st.sidebar.subheader(group)
    for k in keys:
        if k not in features:
            continue
        label, lo, hi, default, step = INPUT_SPEC[k]
        params[k] = st.sidebar.number_input(label, min_value=float(lo),
                                             max_value=float(hi),
                                             value=float(default),
                                             step=float(step))

predict = st.sidebar.button("🔮 Predict properties", use_container_width=True,
                            type="primary")

# ---------------------------------------------------------------- predict
st.subheader("Predicted mechanical properties")

if predict:
    X = pd.DataFrame([{f: params[f] for f in features}])[features].values
    preds = {t: float(models[t]["model"].predict(X)[0]) for t in C.OUTPUTS}

    accept = (preds["YS"] <= C.YS_MAX
              and C.RATIO_MIN <= preds["Ratio"] <= C.RATIO_MAX)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yield Strength", f"{preds['YS']:.0f} MPa",
              delta=f"limit ≤ {C.YS_MAX:.0f}",
              delta_color="normal" if preds["YS"] <= C.YS_MAX else "inverse")
    c2.metric("UTS", f"{preds['UTS']:.0f} MPa")
    c3.metric("% Elongation", f"{preds['Elong']:.1f} %")
    ratio_ok = C.RATIO_MIN <= preds["Ratio"] <= C.RATIO_MAX
    c4.metric("UTS/YS ratio", f"{preds['Ratio']:.3f}",
              delta=f"target {C.RATIO_MIN}–{C.RATIO_MAX}",
              delta_color="normal" if ratio_ok else "inverse")

    st.divider()
    if accept:
        st.success("✅ **ACCEPTABLE** — meets BHEL/L&T requirement "
                   f"(YS ≤ {C.YS_MAX:.0f} MPa and "
                   f"{C.RATIO_MIN} ≤ UTS/YS ≤ {C.RATIO_MAX}).")
    else:
        reasons = []
        if preds["YS"] > C.YS_MAX:
            reasons.append(f"YS {preds['YS']:.0f} MPa exceeds {C.YS_MAX:.0f} MPa")
        if not ratio_ok:
            reasons.append(f"UTS/YS {preds['Ratio']:.3f} outside "
                           f"{C.RATIO_MIN}–{C.RATIO_MAX}")
        st.error("❌ **NOT ACCEPTABLE** — " + "; ".join(reasons) + ".")

    with st.expander("Show input parameters used"):
        st.dataframe(pd.DataFrame([params]).T.rename(columns={0: "value"}))
else:
    st.info("Set the parameters in the sidebar and click **Predict properties**.")

st.caption("Model: per-property regressor selected by cross-validated R² in "
           "train_models.py. Predictions are estimates — confirm critical "
           "heats by physical testing.")
