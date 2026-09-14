import sys
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from modules.ml_engine import PredictorML
from modules.feature_engineering import clean_history

GOAL_LINES = [1.5, 2.5, 3.5]
CORNER_LINES = [8.5, 9.5, 10.5, 11.5, 12.5]
CARD_LINES = [3.5, 4.5, 5.5, 6.5]
CONF_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _selective_binary_metrics(df, p_col, y_col, threshold=0.60):
    p = df[p_col].astype(float)
    y = df[y_col].astype(int)
    sel_over = p >= threshold
    sel_under = p <= (1.0 - threshold)
    selected = sel_over | sel_under
    if not selected.any():
        return {"n": 0, "accuracy": None, "avg_conf": None, "calibration_gap_pp": None}
    pred = np.where(sel_over[selected], 1, 0)
    actual = y[selected].to_numpy()
    conf = np.where(sel_over[selected], p[selected], 1.0 - p[selected])
    acc = float(np.mean(pred == actual))
    avg_conf = float(np.mean(conf))
    return {
        "n": int(selected.sum()),
        "accuracy": acc,
        "avg_conf": avg_conf,
        "calibration_gap_pp": float((avg_conf - acc) * 100.0),
    }


def _calibration_bins(df, p_col, y_col):
    p = df[p_col].astype(float)
    y = df[y_col].astype(int)
    bins = []
    edges = np.arange(0.50, 0.851, 0.05)
    for lo in edges[:-1]:
        hi = lo + 0.05
        mask_hi = (p >= lo) & (p < hi)
        mask_lo = (p <= 1.0 - lo) & (p > 1.0 - hi)
        selected = mask_hi | mask_lo
        if not selected.any():
            continue
        chosen_conf = np.where(mask_hi[selected], p[selected], 1.0 - p[selected])
        chosen_pred = np.where(mask_hi[selected], 1, 0)
        actual = y[selected].to_numpy()
        bins.append({
            "bucket": f"{int(lo*100)}-{int(hi*100)}%",
            "n": int(selected.sum()),
            "avg_conf": round(float(np.mean(chosen_conf)), 4),
            "accuracy": round(float(np.mean(chosen_pred == actual)), 4),
            "gap_pp": round(float((np.mean(chosen_conf) - np.mean(chosen_pred == actual)) * 100), 2),
        })
    return bins


def _line_key(prefix, line):
    return f"{prefix}_{str(line).replace('.', '_')}"


def walk_forward(csv_path, min_train=800, step=150):
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
            base = {
                "Fecha": r.Fecha,
                "Local": r.Local,
                "Visitante": r.Visitante,
                "y": int(r.Target_1X2),
                "Total_Goles": float(r.Total_Goles),
                "Total_Corners": float(r.Total_Corners),
                "Total_Tarjetas": float(r.Total_Tarjetas),
            }

            try:
                pbase = ml.predecir_fila_preparada(r, 2.5, 9.5, 4.5)
            except Exception:
                continue

            base.update({
                "pA": pbase["Resultado_1X2"]["Gana Visita"] / 100.0,
                "pD": pbase["Resultado_1X2"]["Empate"] / 100.0,
                "pH": pbase["Resultado_1X2"]["Gana Local"] / 100.0,
            })

            for line in GOAL_LINES:
                p = ml.predecir_fila_preparada(r, line, 9.5, 4.5)
                k = _line_key("g", line)
                base[f"p_{k}"] = p["Goles_Over_Under"][f"Over {line}"] / 100.0
                base[f"y_{k}"] = int(r.Total_Goles > line)

            for line in CORNER_LINES:
                p = ml.predecir_fila_preparada(r, 2.5, line, 4.5)
                k = _line_key("c", line)
                base[f"p_{k}"] = p["Corners_Totales"][f"Over {line} Corners"] / 100.0
                base[f"y_{k}"] = int(r.Total_Corners > line)

            for line in CARD_LINES:
                p = ml.predecir_fila_preparada(r, 2.5, 9.5, line)
                k = _line_key("t", line)
                base[f"p_{k}"] = p["Tarjetas_Totales"][f"Over {line} Tarjetas"] / 100.0
                base[f"y_{k}"] = int(r.Total_Tarjetas > line)

            rows.append(base)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}

    probs = out[["pA", "pD", "pH"]].to_numpy()
    y = out.y.to_numpy()
    metrics = {
        "n": len(out),
        "logloss_1x2": float(log_loss(y, probs, labels=[0, 1, 2])),
        "accuracy_1x2": float(np.mean(np.argmax(probs, axis=1) == y)),
        "markets": {},
    }

    groups = [
        ("goals", GOAL_LINES, "g"),
        ("corners", CORNER_LINES, "c"),
        ("cards", CARD_LINES, "t"),
    ]
    for market, lines, prefix in groups:
        metrics["markets"][market] = {}
        for line in lines:
            k = _line_key(prefix, line)
            pcol, ycol = f"p_{k}", f"y_{k}"
            info = {
                "brier": float(brier_score_loss(out[ycol], out[pcol])),
                "calibration_bins": _calibration_bins(out, pcol, ycol),
                "selective": {},
            }
            for threshold in CONF_THRESHOLDS:
                info["selective"][str(int(threshold * 100))] = _selective_binary_metrics(
                    out, pcol, ycol, threshold
                )
            metrics["markets"][market][str(line)] = info

    return out, metrics


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/historico_ligamx_completo.csv"
    rows, metrics = walk_forward(path)
    print(metrics)
    rows.to_csv("backtest_predictions_multiline.csv", index=False)
