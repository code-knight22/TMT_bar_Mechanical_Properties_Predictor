"""
Stage 2 — Correlation analysis.

Produces:
  outputs/correlation_matrix.csv          full Pearson matrix
  outputs/correlation_input_output.csv    ranked input->output correlations
  outputs/correlation_heatmap.png         annotated heatmap
  outputs/scatter_top_drivers.png         scatter of strongest driver per output

Run:
    python correlation_analysis.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import config as C
from data_prep import get_clean_data


def run():
    df = get_clean_data()

    # ---- full Pearson matrix ------------------------------------------
    corr = df[C.INPUTS + C.OUTPUTS].corr(method="pearson")
    corr.to_csv(C.OUTPUT_DIR / "correlation_matrix.csv")

    # ---- ranked input -> output correlations --------------------------
    rows = []
    for out in C.OUTPUTS:
        for inp in C.INPUTS:
            r, p = pearsonr(df[inp], df[out])
            s, _ = spearmanr(df[inp], df[out])
            rows.append([out, inp, round(r, 3), round(p, 4), round(s, 3)])
    table = pd.DataFrame(rows, columns=["Output", "Input",
                                        "Pearson_r", "p_value", "Spearman_r"])
    table.to_csv(C.OUTPUT_DIR / "correlation_input_output.csv", index=False)

    print("=== Stage 2: Correlation analysis ===")
    for out in C.OUTPUTS:
        sub = table[table.Output == out].reindex(
            table[table.Output == out].Pearson_r.abs().sort_values(ascending=False).index)
        print(f"\nTop drivers for {out}:")
        print(sub[["Input", "Pearson_r", "p_value", "Spearman_r"]].head(5).to_string(index=False))

    # ---- multicollinearity among inputs -------------------------------
    print("\nStrong input-input correlations (|r| > 0.6):")
    ic = df[C.INPUTS].corr()
    for i in range(len(C.INPUTS)):
        for j in range(i + 1, len(C.INPUTS)):
            if abs(ic.iloc[i, j]) > 0.6:
                print(f"  {C.INPUTS[i]} - {C.INPUTS[j]}: {ic.iloc[i, j]:.2f}")

    # ---- heatmap ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=9)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "black", fontsize=7)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(f"Correlation matrix — Fe550 25mm MM TMT (n={len(df)})", pad=12)
    plt.tight_layout()
    plt.savefig(C.OUTPUT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- scatter of the strongest driver per output ------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, out in zip(axes.ravel(), C.OUTPUTS):
        sub = table[table.Output == out]
        drv = sub.loc[sub.Pearson_r.abs().idxmax(), "Input"]
        ax.scatter(df[drv], df[out], s=12, alpha=0.5)
        ax.set_xlabel(C.LABELS[drv]); ax.set_ylabel(C.LABELS[out])
        r = sub.loc[sub.Input == drv, "Pearson_r"].values[0]
        ax.set_title(f"{out} vs {drv}  (r={r})")
    plt.suptitle("Strongest linear driver per property", y=1.01)
    plt.tight_layout()
    plt.savefig(C.OUTPUT_DIR / "scatter_top_drivers.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved correlation outputs to: {C.OUTPUT_DIR}")
    return table


if __name__ == "__main__":
    run()
