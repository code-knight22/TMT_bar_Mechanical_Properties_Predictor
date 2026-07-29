"""
Central configuration for the Fe550 TMT predictive-model project.
Edit paths / column names here if the source file changes.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
# Resolve paths relative to THIS file's folder so the app works no matter how
# the repo is laid out (project/code/... locally, or code contents at the repo
# root on Streamlit Cloud). models/ and outputs/ always live next to this file.
CODE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

# Training data (only used by data_prep.py / train_models.py): look next to the
# code first, then one level up in the project root.
_data_name  = "Data for Corelation.xlsx"
DATA_FILE   = CODE_DIR / _data_name if (CODE_DIR / _data_name).exists() \
              else PROJECT_ROOT / _data_name
SHEET_NAME  = "Sheet2"
OUTPUT_DIR  = CODE_DIR / "outputs"       # plots, csv, metrics
MODEL_DIR   = CODE_DIR / "models"        # saved .joblib models  (commit these)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------- data structure
# In the raw sheet the real header is on row 4 (1-based) as A..N and the
# engineering names are on the row above. We read by position and rename.
# Column order in the sheet: A..N  ->  the 14 names below.
COLUMN_NAMES = [
    "C", "Mn", "CE", "RPM", "LinSpeed", "T_11std", "WaterFlow",
    "WaterPress", "T_7pipe", "EqualTemp", "YS", "UTS", "Elong", "Ratio",
]
# Row index (0-based) of the first data row inside the sheet.
FIRST_DATA_ROW = 5

INPUTS  = ["C", "Mn", "CE", "RPM", "LinSpeed", "T_11std",
           "WaterFlow", "WaterPress", "T_7pipe", "EqualTemp"]
OUTPUTS = ["YS", "UTS", "Elong", "Ratio"]

# Human-readable labels for reports / plots
LABELS = {
    "C": "%C", "Mn": "%Mn", "CE": "Carbon Equiv. (C+Mn/6)",
    "RPM": "12th-stand RPM", "LinSpeed": "Linear speed (m/s)",
    "T_11std": "11th-stand temp (°C)", "WaterFlow": "Water flow (Nm³/hr)",
    "WaterPress": "Water pressure (kg/cm²)", "T_7pipe": "Temp after 7th pipe (°C)",
    "EqualTemp": "Equalisation temp (°C)",
    "YS": "Yield Strength (MPa)", "UTS": "UTS (MPa)",
    "Elong": "% Elongation", "Ratio": "UTS/YS ratio",
}

# --------------------------------------------- customer acceptance rules
# BHEL / L&T requirement for Fe550:
YS_MAX      = 660.0        # MPa
RATIO_MIN   = 1.15
RATIO_MAX   = 1.25

RANDOM_STATE = 42
