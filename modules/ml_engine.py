import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .feature_engineering import (
    add_rolling_features,
    build_current_team_feature_cache,
    current_match_features,
    normalize_team,
)


class PredictorML:
    """Modelo ML V3: solo información prepartido y mayor peso a datos recientes."""

    def __init__(self, random_state=42, training_half_life_days=365.0):
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
        self.training_half_life_days = float(training_half_life_days)
        self.current_team_cache = None

    def preparar_dataset(self, df):
        d = add_rolling_features(df)
        prefixes = ("H_", "A_", "Diff_")
        self.features = ["Diff_ELO_Pre"] + [
            c for c in d.columns
            if c.startswith(prefixes)
            and c not in {"Diff_ELO_Pre"}
            and not c.startswith(("H_idx", "A_idx"))
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
            "Fecha", "Target_1X2", "Total_Goles", "Total_Corners", "Total_Tarjetas"
        ]
        return d.dropna(subset=required).reset_index(drop=True)

    def _prep(self, df):
        return self.preparar_dataset(df)

    def _training_weights(self, dates):
        dates = pd.to_datetime(dates, errors="coerce")
        if dates.isna().all() or self.training_half_life_days <= 0:
            return np.ones(len(dates), dtype=float)
        latest = dates.max()
        age_days = (latest - dates).dt.days.clip(lower=0).astype(float)
        weights = np.power(0.5, age_days / self.training_half_life_days)
        return np.clip(weights.to_numpy(dtype=float), 0.10, 1.0)

    def set_current_context(self, df_historico):
        """Precalcula forma vigente una vez; evita rehacer todo el histórico por fixture."""
        self.current_team_cache = build_current_team_feature_cache(df_historico)
        return self.current_team_cache

    def entrenar_preparado(self, d):
        if d is None or len(d) < 300:
            return False
        if not self.features:
            prefixes = ("H_", "A_", "Diff_")
            self.features = ["Diff_ELO_Pre"] + [
                c for c in d.columns
                if c.startswith(prefixes)
                and c not in {"Diff_ELO_Pre"}
                and not c.startswith(("H_idx", "A_idx"))
            ]
            self.features = [c for c in self.features if pd.api.types.is_numeric_dtype(d[c])]

        split = max(int(len(d) * 0.8), len(d) - 300)
        tr, cal = d.iloc[:split], d.iloc[split:]
        if len(cal) < 50:
            return False

        X = tr[self.features]
        sample_weight = self._training_weights(tr["Fecha"])
        self.model_1x2.fit(X, tr.Target_1X2, sample_weight=sample_weight)
        self.reg_goles.fit(X, tr.Total_Goles, sample_weight=sample_weight)
        self.reg_corners.fit(X, tr.Total_Corners, sample_weight=sample_weight)
        self.reg_cards.fit(X, tr.Total_Tarjetas, sample_weight=sample_weight)

        Xc = cal[self.features]
        self.resid_g = (cal.Total_Goles - self.reg_goles.predict(Xc)).to_numpy()
        self.resid_c = (cal.Total_Corners - self.reg_corners.predict(Xc)).to_numpy()
        self.resid_t = (cal.Total_Tarjetas - self.reg_cards.predict(Xc)).to_numpy()
        self.is_trained = True
        return True

    def entrenar(self, df_historico):
        ok = self.entrenar_preparado(self.preparar_dataset(df_historico))
        if ok:
            self.set_current_context(df_historico)
        return ok

    @staticmethod
    def _asian_side_probs(draws, line, side):
        draws = np.asarray(draws)
        line = float(line)
        side = str(side).lower()
        frac = round(line - np.floor(line), 2)

        if frac in (0.25, 0.75):
            if frac == 0.25:
                n = int(np.floor(line))
                if side == "over":
                    full_win = draws >= n + 1
                    half_win = np.zeros_like(draws, dtype=bool)
                    push = np.zeros_like(draws, dtype=bool)
                    half_loss = draws == n
                    full_loss = draws <= n - 1
                else:
                    full_win = draws <= n - 1
                    half_win = draws == n
                    push = np.zeros_like(draws, dtype=bool)
                    half_loss = np.zeros_like(draws, dtype=bool)
                    full_loss = draws >= n + 1
            else:
                n = int(np.floor(line))
                split_int = n + 1
                if side == "over":
                    full_win = draws >= split_int + 1
                    half_win = draws == split_int
                    push = np.zeros_like(draws, dtype=bool)
                    half_loss = np.zeros_like(draws, dtype=bool)
                    full_loss = draws <= n
                else:
                    full_win = draws <= n
                    half_win = np.zeros_like(draws, dtype=bool)
                    push = np.zeros_like(draws, dtype=bool)
                    half_loss = draws == split_int
                    full_loss = draws >= split_int + 1
        else:
            if side == "over":
                full_win = draws > line
                full_loss = draws < line
            else:
                full_win = draws < line
                full_loss = draws > line
            push = draws == line if line.is_integer() else np.zeros_like(draws, dtype=bool)
            half_win = np.zeros_like(draws, dtype=bool)
            half_loss = np.zeros_like(draws, dtype=bool)

        pct = lambda mask: round(float(np.mean(mask) * 100.0), 1)
        return {
            "win": pct(full_win),
            "half_win": pct(half_win),
            "push": pct(push),
            "half_loss": pct(half_loss),
            "loss": pct(full_loss),
        }

    @classmethod
    def _market_payload(cls, pred, line, resid, suffix=""):
        if resid.size < 50:
            raise ValueError("Muestra de calibracion insuficiente: NO BET")
        draws = np.clip(np.rint(float(pred) + resid), 0, None)
        over = cls._asian_side_probs(draws, line, "over")
        under = cls._asian_side_probs(draws, line, "under")
        o = f"Over {line}{suffix}"
        u = f"Under {line}{suffix}"
        p = f"Push {line}{suffix}"
        return {
            o: over["win"],
            u: under["win"],
            p: max(over["push"], under["push"]),
            f"HalfWin {o}": over["half_win"],
            f"HalfLoss {o}": over["half_loss"],
            f"Loss {o}": over["loss"],
            f"HalfWin {u}": under["half_win"],
            f"HalfLoss {u}": under["half_loss"],
            f"Loss {u}": under["loss"],
        }

    def _predict_X(self, X, linea_goles, linea_corners, linea_tarjetas):
        probs = dict(zip(self.model_1x2.classes_, self.model_1x2.predict_proba(X)[0]))
        pg = float(self.reg_goles.predict(X)[0])
        pc = float(self.reg_corners.predict(X)[0])
        pt = float(self.reg_cards.predict(X)[0])

        return {
            "Resultado_1X2": {
                "Gana Local": round(probs.get(2, 0) * 100, 1),
                "Empate": round(probs.get(1, 0) * 100, 1),
                "Gana Visita": round(probs.get(0, 0) * 100, 1),
            },
            "Goles_Over_Under": self._market_payload(pg, linea_goles, self.resid_g),
            "Corners_Totales": self._market_payload(pc, linea_corners, self.resid_c, " Corners"),
            "Tarjetas_Totales": self._market_payload(pt, linea_tarjetas, self.resid_t, " Tarjetas"),
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
            df_historico,
            normalize_team(equipo_local),
            normalize_team(equipo_visita),
            team_cache=self.current_team_cache,
            elo_local=elo_local,
            elo_visita=elo_visita,
        )
        return self.predecir_fila_preparada(f, linea_goles, linea_corners, linea_tarjetas)
