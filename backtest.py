import sys
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from modules.ml_engine import PredictorML
from modules.feature_engineering import clean_history


def walk_forward(csv_path, min_train=800, step=75):
    """Backtest cronologico por bloques.

    El modelo se reentrena al inicio de cada bloque. Dentro del bloque, las
    features prepartido si incorporan resultados previos ya ocurridos del mismo
    bloque, evitando usar informacion futura y evitando que la forma quede
    congelada durante todo el bloque.
    """
    df = clean_history(pd.read_csv(csv_path))
    rows = []

    for end in range(min_train, len(df), step):
        train = df.iloc[:end].copy()
        test = df.iloc[end:min(end + step, len(df))].copy()

        ml = PredictorML()
        if not ml.entrenar(train):
            continue

        context = train.copy()
        for _, r in test.iterrows():
            try:
                p = ml.predecir_mercados_completos(
                    context,
                    r.Local,
                    r.Visitante,
                    linea_goles=2.5,
                    linea_corners=9.5,
                    linea_tarjetas=4.5,
                )
            except Exception:
                context = pd.concat([context, pd.DataFrame([r])], ignore_index=True)
                continue

            y = 2 if r.Goles_L > r.Goles_V else 0 if r.Goles_L < r.Goles_V else 1
            rows.append({
                "Fecha": r.Fecha,
                "Local": r.Local,
                "Visitante": r.Visitante,
                "y": y,
                "pA": p["Resultado_1X2"]["Gana Visita"] / 100.0,
                "pD": p["Resultado_1X2"]["Empate"] / 100.0,
                "pH": p["Resultado_1X2"]["Gana Local"] / 100.0,
                "og25": p["Goles_Over_Under"]["Over 2.5"] / 100.0,
                "yog25": int(r.Goles_L + r.Goles_V > 2.5),
                "oc95": p["Corners_Totales"]["Over 9.5 Corners"] / 100.0,
                "yoc95": int(r.Corners_L + r.Corners_V > 9.5),
                "ot45": p["Tarjetas_Totales"]["Over 4.5 Tarjetas"] / 100.0,
                "yot45": int(
                    r.Amarillas_L + 2 * r.Rojas_L +
                    r.Amarillas_V + 2 * r.Rojas_V > 4.5
                ),
            })

            # A partir del siguiente partido este resultado ya pertenece al pasado.
            context = pd.concat([context, pd.DataFrame([r])], ignore_index=True)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}

    probs = out[["pA", "pD", "pH"]].to_numpy()
    y = out.y.to_numpy()

    metrics = {
        "n": len(out),
        "logloss_1x2": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier_goals_o25": float(brier_score_loss(out.yog25, out.og25)),
        "brier_corners_o95": float(brier_score_loss(out.yoc95, out.oc95)),
        "brier_cards_o45": float(brier_score_loss(out.yot45, out.ot45)),
        "accuracy_1x2": float(np.mean(np.argmax(probs, axis=1) == y)),
    }
    return out, metrics


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/historico_ligamx_completo.csv"
    rows, metrics = walk_forward(path)
    print(metrics)
    rows.to_csv("backtest_predictions.csv", index=False)
