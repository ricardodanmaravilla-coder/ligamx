import os
import pandas as pd

from modules.feature_engineering import clean_history
from modules.ml_engine import PredictorML
from modules.parquet_cache import HISTORY_PARQUET, FEATURES_PARQUET, HISTORY_CSV_CANDIDATES


def main():
    csv_path = next((p for p in HISTORY_CSV_CANDIDATES if os.path.exists(p)), None)
    if not csv_path:
        raise SystemExit("No se encontró histórico CSV para construir Parquet")

    os.makedirs(os.path.dirname(HISTORY_PARQUET), exist_ok=True)
    df = clean_history(pd.read_csv(csv_path))
    df.to_parquet(HISTORY_PARQUET, index=False, compression="zstd")

    ml = PredictorML()
    prepared = ml.preparar_dataset(df)
    prepared.to_parquet(FEATURES_PARQUET, index=False, compression="zstd")

    print(
        f"Parquet cache OK: history={len(df)} rows, "
        f"features={len(prepared)} rows, feature_cols={len(ml.features)}"
    )


if __name__ == "__main__":
    main()
