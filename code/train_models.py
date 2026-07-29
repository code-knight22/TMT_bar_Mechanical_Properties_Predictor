"""
Stage 3-6 — Feature selection, model training, testing and evaluation.

WHY RANDOM FOREST IS THE PRIMARY MODEL
--------------------------------------
The correlation analysis shows that no single process/chemistry parameter has a
strong LINEAR relationship with the tensile properties (all |Pearson r| <= ~0.2).
Yet metallurgy tells us the properties ARE controlled by these parameters — the
dependence is non-linear and driven by INTERACTIONS (e.g. quench water flow only
matters at a given rolling temperature and bar section). A model class is needed
that:
    * captures non-linearity and feature interactions automatically,
    * needs no feature scaling and tolerates different units,
    * is robust to outliers and modest noise in shop-floor data,
    * gives interpretable feature-importance for process engineers,
    * trains reliably on a few-hundred-row dataset without overfitting.
A Random Forest regressor satisfies all of these. We still TRAIN two baselines
(Linear Regression, Gradient Boosting) and pick the winner per property by
cross-validated R2, so the choice is justified by evidence, not assertion.

Run:
    python train_models.py
"""
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import config as C
from data_prep import get_clean_data


# ------------------------------------------------------------------ models
def candidate_models(random_state=C.RANDOM_STATE):
    """Return the model zoo. Linear is wrapped in a scaler pipeline."""
    return {
        "LinearRegression": Pipeline([
            ("scale", StandardScaler()),
            ("lr", LinearRegression()),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=3,
            max_features="sqrt", n_jobs=-1, random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=random_state),
    }


def evaluate_metrics(y_true, y_pred):
    return {
        "R2": round(float(r2_score(y_true, y_pred)), 3),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
    }


# ------------------------------------------------------------------ main
def run():
    df = get_clean_data()

    # --- Stage 3: feature selection ------------------------------------
    # CE = C + Mn/6 is collinear with C and Mn, so we drop it to avoid
    # redundancy. All remaining process/chemistry variables are retained;
    # the Random Forest will down-weight the uninformative ones itself.
    features = [f for f in C.INPUTS if f != "CE"]
    print("=== Stage 3: Feature selection ===")
    print("  Dropped 'CE' (collinear: CE = C + Mn/6).")
    print("  Using features:", features)

    X = df[features].values
    kf = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)

    cv_results, best_models, final_metrics, holdout_pred = {}, {}, {}, {}

    for target in C.OUTPUTS:
        y = df[target].values

        # --- Stage 4: model comparison by 5-fold CV R2 ----------------
        cv_results[target] = {}
        for name, model in candidate_models().items():
            scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
            cv_results[target][name] = round(float(scores.mean()), 3)

        winner = max(cv_results[target], key=cv_results[target].get)

        # --- Stage 5: train winner, test on 20% hold-out --------------
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=C.RANDOM_STATE)
        model = candidate_models()[winner]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        final_metrics[target] = {"selected_model": winner,
                                 "holdout": evaluate_metrics(y_te, pred)}
        holdout_pred[target] = (y_te, pred)

        # refit on ALL data for deployment and save
        model_full = candidate_models()[winner]
        model_full.fit(X, y)
        joblib.dump({"model": model_full, "features": features},
                    C.MODEL_DIR / f"model_{target}.joblib")
        best_models[target] = model_full

        print(f"\n--- {target} ---")
        print("  CV R2:", cv_results[target], "-> selected:", winner)
        print("  Hold-out:", final_metrics[target]["holdout"])

    # --- Stage 6: acceptance evaluation & feature importance ----------
    ys_a, ys_p = holdout_pred["YS"]
    r_a, r_p = holdout_pred["Ratio"]
    act_ok = (ys_a <= C.YS_MAX) & (r_a >= C.RATIO_MIN) & (r_a <= C.RATIO_MAX)
    pred_ok = (ys_p <= C.YS_MAX) & (r_p >= C.RATIO_MIN) & (r_p <= C.RATIO_MAX)
    acceptance = {
        "actual_accept_rate": round(float(act_ok.mean()), 3),
        "predicted_accept_rate": round(float(pred_ok.mean()), 3),
        "accept_classifier_accuracy": round(float((act_ok == pred_ok).mean()), 3),
    }
    print("\n=== Stage 6: Acceptance prediction (YS<=660 & 1.15<=UTS/YS<=1.25) ===")
    print(" ", acceptance)

    # feature importance from the tree models (skip if a linear model won)
    importances = {}
    for target, model in best_models.items():
        if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
            importances[target] = dict(zip(features,
                                            np.round(model.feature_importances_, 4)))
    if importances:
        imp_df = pd.DataFrame(importances)
        imp_df.to_csv(C.OUTPUT_DIR / "feature_importance.csv")
        ax = imp_df.mean(axis=1).sort_values().plot.barh(
            figsize=(9, 5), color="steelblue")
        ax.set_title("Mean feature importance (tree models, across targets)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(C.OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

    # predicted-vs-actual plots
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, target in zip(axes.ravel(), C.OUTPUTS):
        a, p = holdout_pred[target]
        ax.scatter(a, p, s=12, alpha=0.5)
        lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        m = final_metrics[target]["holdout"]
        ax.set_xlabel(f"Actual {target}"); ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{target}  R2={m['R2']}  MAE={m['MAE']}")
    plt.suptitle("Predicted vs actual (20% hold-out)", y=1.01)
    plt.tight_layout()
    plt.savefig(C.OUTPUT_DIR / "pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()

    # persist all metrics
    summary = {"cv_r2": cv_results, "final": final_metrics,
               "acceptance": acceptance, "features": features}
    json.dump(summary, open(C.OUTPUT_DIR / "metrics_summary.json", "w"), indent=2)
    print(f"\nSaved models -> {C.MODEL_DIR}")
    print(f"Saved metrics & plots -> {C.OUTPUT_DIR}")


if __name__ == "__main__":
    run()
