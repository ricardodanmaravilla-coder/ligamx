import os
import pandas as pd

from .feature_engineering import clean_history
from .mc_context import context_to_map

HISTORY_PARQUET = "data/historico_ligamx_completo.parquet"
FEATURES_PARQUET = "data/ligamx_features_v3.parquet"
MC_CONTEXT_PARQUET = "data/ligamx_mc_context_v1.parquet"
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


def load_mc_context():
    """Contexto estadístico precalculado para Monte Carlo."""
    if not os.path.exists(MC_CONTEXT_PARQUET):
        return None
    return context_to_map(pd.read_parquet(MC_CONTEXT_PARQUET))


def cache_info():
    return {
        "history_parquet": os.path.exists(HISTORY_PARQUET),
        "features_parquet": os.path.exists(FEATURES_PARQUET),
        "mc_context_parquet": os.path.exists(MC_CONTEXT_PARQUET),
        "history_path": HISTORY_PARQUET,
        "features_path": FEATURES_PARQUET,
        "mc_context_path": MC_CONTEXT_PARQUET,
    }
