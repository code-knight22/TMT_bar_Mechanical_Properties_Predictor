"""
Inference helper — predict YS, UTS, %Elongation and UTS/YS ratio for a new
set of process/chemistry parameters, and report whether the bar is expected
to meet the BHEL/L&T acceptance window.

Usage:
    from predict import predict_properties
    predict_properties({
        "C": 0.21, "Mn": 0.68, "RPM": 558, "LinSpeed": 9.36,
        "T_11std": 1121, "WaterFlow": 792, "WaterPress": 13.6,
        "T_7pipe": 39, "EqualTemp": 521,
    })

Run `python predict.py` for a demo on one example row.
"""
import joblib
import pandas as pd

import config as C


def _load_models():
    models = {}
    for target in C.OUTPUTS:
        path = C.MODEL_DIR / f"model_{target}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run train_models.py first.")
        models[target] = joblib.load(path)
    return models


def predict_properties(params: dict) -> dict:
    """params: dict of the input features (CE is optional / ignored)."""
    models = _load_models()
    feats = models["YS"]["features"]
    missing = [f for f in feats if f not in params]
    if missing:
        raise ValueError(f"Missing input parameters: {missing}")

    X = pd.DataFrame([{f: params[f] for f in feats}])[feats].values
    preds = {t: round(float(models[t]["model"].predict(X)[0]), 2)
             for t in C.OUTPUTS}

    accept = (preds["YS"] <= C.YS_MAX
              and C.RATIO_MIN <= preds["Ratio"] <= C.RATIO_MAX)
    preds["acceptable"] = bool(accept)
    preds["acceptance_rule"] = (
        f"YS<={C.YS_MAX} MPa and {C.RATIO_MIN}<=UTS/YS<={C.RATIO_MAX}")
    return preds


if __name__ == "__main__":
    demo = {
        "C": 0.21, "Mn": 0.68, "RPM": 558, "LinSpeed": 9.36,
        "T_11std": 1121, "WaterFlow": 792, "WaterPress": 13.6,
        "T_7pipe": 39, "EqualTemp": 521,
    }
    print("Input:", demo)
    print("Prediction:", predict_properties(demo))
