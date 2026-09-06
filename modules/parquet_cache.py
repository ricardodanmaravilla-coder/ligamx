import os
import pandas as pd

from .feature_engineering import clean_history

HISTORY_PARQUET = "data/historico_ligamx_completo.parquet"
FEATURES_PARQUET = "data/ligamx_features_v3.parquet"
HISTORY_CSV_CANDIDATES = (
    "data/historico_ligamx_completo.csv",
    "historico_ligamx_completo.csv",
)


def load_history_fast():
    """Carga Parquet si existe; CSV queda solo como fallback seguro."""
    if os.path.exists(HISTORY_PARQUET):
        return clean_history(pd.read_parquet(HISTORY_PARQUET))
    for path in HISTORY_CSV_CANDIDATES:
        if os.path.exists(path):
            return clean_history(pd.read_csv(path))
    raise RuntimeError("No se pudo cargar el histórico de Liga MX")


def load_prepared_features():
    """Dataset ML ya feature-engineered durante el build de la imagen."""
    if not os.path.exists(FEATURES_PARQUET):
        return None
    d = pd.read_parquet(FEATURES_PARQUET)
    if "Fecha" in d.columns:
        d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
        d = d.sort_values("Fecha").reset_index(drop=True)
    return d


def cache_info():
    return {
        "history_parquet": os.path.exists(HISTORY_PARQUET),
        "features_parquet": os.path.exists(FEATURES_PARQUET),
        "history_path": HISTORY_PARQUET,
        "features_path": FEATURES_PARQUET,
    }
