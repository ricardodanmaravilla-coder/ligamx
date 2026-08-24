import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .feature_engineering import add_rolling_features, current_match_features, normalize_team


class PredictorML:
    """Modelo ML V2 entrenado solo con informacion disponible antes del partido."""

    def __init__(self, random_state=42):
        self.model_1x2 = RandomForestClassifier(
            n_estimators=350, min_samples_leaf=8, max_features="sqrt",
            class_weight="balanced_subsample", random_state=random_state, n_jobs=-1
        )
        self.reg_goles = RandomForestRegressor(
            n_estimators=350, min_samples_leaf=8, max_features=0.7,
            random_state=random_state, n_jobs=-1
        )
        self.reg_corners = RandomForestRegressor(
            n_estimators=350, min_samples_leaf=8, max_features=0.7,
            random_state=random_state, n_jobs=-1
        )
        self.reg_cards = RandomForestRegressor(
            n_estimators=350, min_samples_leaf=8, max_features=0.7,
            random_state=random_state, n_jobs=-1
        )
        self.features = []
        self.resid_g = np.array([])
        self.resid_c = np.array([])
        self.resid_t = np.array([])
        self.is_trained = False

    def preparar_dataset(self, df):
        d = add_rolling_features(df)
        prefixes = ("H_", "A_")
        self.features = ["Diff_ELO_Pre"] + [
            c for c in d.columns
            if c.startswith(prefixes) and not c.startswith(("H_idx", "A_idx"))
        ]
        self.features = [c for c in self.features if pd.api.types.is_numeric_dtype(d[c])]

        d["Target_1X2"] = np.where(
            d.Goles_L > d.Goles_V, 2,
            np.where(d.Goles_L < d.Goles_V, 0, 1)
        )
        d["Total_Goles"] = d.Goles_L + d.Goles_V
        d["Total_Corners"] = d.Corners_L + d.Corners_V
        d["Total_Tarjetas"] = (
            d.Amarillas_L + 2 * d.Rojas_L + d.Amarillas_V + 2 * d.Rojas_V
        )
        required = self.features + [
            "Target_1X2", "Total_Goles", "Total_Corners", "Total_Tarjetas"
        ]
        return d.dropna(subset=required).reset_index(drop=True)

    # Alias temporal para compatibilidad interna.
    def _prep(self, df):
        return self.preparar_dataset(df)

    def entrenar_preparado(self, d):
        if d is None or len(d) < 300:
            return False
        if not self.features:
            prefixes = ("H_", "A_")
            self.features = ["Diff_ELO_Pre"] + [
                c for c in d.columns
                if c.startswith(prefixes) and not c.startswith(("H_idx", "A_idx"))
            ]
            self.features = [c for c in self.features if pd.api.types.is_numeric_dtype(d[c])]

        split = max(int(len(d) * 0.8), len(d) - 300)
        tr, cal = d.iloc[:split], d.iloc[split:]
        if len(cal) < 50:
            return False

        X = tr[self.features]
        self.model_1x2.fit(X, tr.Target_1X2)
        self.reg_goles.fit(X, tr.Total_Goles)
        self.reg_corners.fit(X, tr.Total_Corners)
        self.reg_cards.fit(X, tr.Total_Tarjetas)

        Xc = cal[self.features]
        self.resid_g = (cal.Total_Goles - self.reg_goles.predict(Xc)).to_numpy()
        self.resid_c = (cal.Total_Corners - self.reg_corners.predict(Xc)).to_numpy()
        self.resid_t = (cal.Total_Tarjetas - self.reg_cards.predict(Xc)).to_numpy()
        self.is_trained = True
        return True

    def entrenar(self, df_historico):
        return self.entrenar_preparado(self.preparar_dataset(df_historico))

    @staticmethod
    def _market_probs(pred, line, resid):
        if resid.size < 50:
            raise ValueError("Muestra de calibracion insuficiente: NO BET")
        line = float(line)
        draws = np.clip(np.rint(float(pred) + resid), 0, None)
        over = float(np.mean(draws > line) * 100.0)
        push = float(np.mean(draws == line) * 100.0) if line.is_integer() else 0.0
        under = max(0.0, 100.0 - over - push)
        return round(over, 1), round(under, 1), round(push, 1)

    def _predict_X(self, X, linea_goles, linea_corners, linea_tarjetas):
        probs = dict(zip(self.model_1x2.classes_, self.model_1x2.predict_proba(X)[0]))
        pg = float(self.reg_goles.predict(X)[0])
        pc = float(self.reg_corners.predict(X)[0])
        pt = float(self.reg_cards.predict(X)[0])

        og, ug, pg_push = self._market_probs(pg, linea_goles, self.resid_g)
        oc, uc, pc_push = self._market_probs(pc, linea_corners, self.resid_c)
        ot, ut, pt_push = self._market_probs(pt, linea_tarjetas, self.resid_t)

        return {
            "Resultado_1X2": {
                "Gana Local": round(probs.get(2, 0) * 100, 1),
                "Empate": round(probs.get(1, 0) * 100, 1),
                "Gana Visita": round(probs.get(0, 0) * 100, 1),
            },
            "Goles_Over_Under": {
                f"Over {linea_goles}": og, f"Under {linea_goles}": ug,
                f"Push {linea_goles}": pg_push,
            },
            "Corners_Totales": {
                f"Over {linea_corners} Corners": oc,
                f"Under {linea_corners} Corners": uc,
                f"Push {linea_corners} Corners": pc_push,
            },
            "Tarjetas_Totales": {
                f"Over {linea_tarjetas} Tarjetas": ot,
                f"Under {linea_tarjetas} Tarjetas": ut,
                f"Push {linea_tarjetas} Tarjetas": pt_push,
            },
            "Prediccion_Totales": {
                "goles": round(pg, 2), "corners": round(pc, 2), "tarjetas": round(pt, 2)
            },
        }

    def predecir_fila_preparada(self, fila, linea_goles=2.5, linea_corners=9.5, linea_tarjetas=4.5):
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        X = pd.DataFrame([{c: fila.get(c, np.nan) for c in self.features}])
        if X.isna().any(axis=None):
            raise ValueError("Features prepartido incompletas: NO BET")
        return self._predict_X(X, linea_goles, linea_corners, linea_tarjetas)

    def predecir_mercados_completos(
        self, df_historico, equipo_local, equipo_visita,
        goles_sim_l=None, goles_sim_v=None, elo_local=None, elo_visita=None,
        linea_goles=2.5, linea_corners=9.5, linea_tarjetas=4.5
    ):
        if not self.is_trained and not self.entrenar(df_historico):
            return {}
        f = current_match_features(
            df_historico, normalize_team(equipo_local), normalize_team(equipo_visita)
        )
        return self.predecir_fila_preparada(f, linea_goles, linea_corners, linea_tarjetas)
