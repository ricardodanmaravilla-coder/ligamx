import sys
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from modules.ml_engine import PredictorML
from modules.feature_engineering import clean_history


def walk_forward(csv_path, min_train=800, step=150):
    """Backtest cronologico eficiente con features prepartido precalculadas.

    add_rolling_features usa shift(1) y ELO prepartido, por lo que cada fila solo
    contiene informacion disponible antes de ese encuentro. El modelo se
    reentrena por bloques y nunca ve targets del bloque de prueba.
    """
    raw = clean_history(pd.read_csv(csv_path))
    prep_builder = PredictorML()
    prepared = prep_builder.preparar_dataset(raw)
    feature_names = list(prep_builder.features)
    rows = []

    for end in range(min_train, len(prepared), step):
        train = prepared.iloc[:end].copy()
        test = prepared.iloc[end:min(end + step, len(prepared))].copy()

        ml = PredictorML()
        ml.features = feature_names
        if not ml.entrenar_preparado(train):
            continue

        for _, r in test.iterrows():
            try:
                p = ml.predecir_fila_preparada(
                    r,
                    linea_goles=2.5,
                    linea_corners=9.5,
                    linea_tarjetas=4.5,
                )
            except Exception:
                continue

            y = int(r.Target_1X2)
            rows.append({
                "Fecha": r.Fecha,
                "Local": r.Local,
                "Visitante": r.Visitante,
                "y": y,
                "pA": p["Resultado_1X2"]["Gana Visita"] / 100.0,
                "pD": p["Resultado_1X2"]["Empate"] / 100.0,
                "pH": p["Resultado_1X2"]["Gana Local"] / 100.0,
                "og25": p["Goles_Over_Under"]["Over 2.5"] / 100.0,
                "yog25": int(r.Total_Goles > 2.5),
                "oc95": p["Corners_Totales"]["Over 9.5 Corners"] / 100.0,
                "yoc95": int(r.Total_Corners > 9.5),
                "ot45": p["Tarjetas_Totales"]["Over 4.5 Tarjetas"] / 100.0,
                "yot45": int(r.Total_Tarjetas > 4.5),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}

    probs = out[["pA", "pD", "pH"]].to_numpy()
    y = out.y.to_numpy()

    metrics = {
        "n": len(out),
        "logloss_1x2": float(log_loss(y, probs, labels=[0, 1, 2])),
        "accuracy_1x2": float(np.mean(np.argmax(probs, axis=1) == y)),
        "brier_goals_o25": float(brier_score_loss(out.yog25, out.og25)),
        "brier_corners_o95": float(brier_score_loss(out.yoc95, out.oc95)),
        "brier_cards_o45": float(brier_score_loss(out.yot45, out.ot45)),
    }
    return out, metrics


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/historico_ligamx_completo.csv"
    rows, metrics = walk_forward(path)
    print(metrics)
    rows.to_csv("backtest_predictions.csv", index=False)
