import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from modules.elo_engine import SistemaEloLigaMX

class PredictorML:
    def __init__(self):
        # 4 Modelos independientes para cada mercado (con parámetros anti-sobreajuste)
        self.model_1x2 = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15, random_state=42)
        self.model_goles = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15, random_state=42)
        self.model_corners = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15, random_state=42)
        self.model_tarjetas = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15, random_state=42)
        
        self.is_trained = False
        self.features = []

    def entrenar(self, df_historico):
        """Entrenamiento independiente con lectura completa de variables."""
        df = df_historico.copy()
        if 'Fecha' in df.columns:
            df = df.sort_values(by='Fecha').reset_index(drop=True)
        
        # 1. Targets Reales (Lo que el modelo intentará adivinar)
        df['Target_1X2'] = df.apply(lambda row: 1 if row['Goles_L'] > row['Goles_V'] else (2 if row['Goles_L'] < row['Goles_V'] else 0), axis=1)
        df['Target_Over_Goles'] = ((df['Goles_L'] + df['Goles_V']) > 2.5).astype(int)
        df['Target_Over_Corners'] = ((df['Corners_L'] + df['Corners_V']) > 9.5).astype(int) if 'Corners_L' in df.columns else 0
        
        # Soporte dinámico para Tarjetas (puede llamarse Tarjetas o Amarillas)
        col_tarj_l = 'Tarjetas_L' if 'Tarjetas_L' in df.columns else 'Amarillas_L'
        col_tarj_v = 'Tarjetas_V' if 'Tarjetas_V' in df.columns else 'Amarillas_V'
        if col_tarj_l in df.columns:
            df['Target_Over_Tarjetas'] = ((df[col_tarj_l] + df[col_tarj_v]) > 4.5).astype(int)
        else:
            df['Target_Over_Tarjetas'] = 0

        # 2. Reconstrucción Histórica del ELO
        motor_elo = SistemaEloLigaMX()
        diff_elo = []
        for index, row in df.iterrows():
            loc, vis = row['Local'], row['Visitante']
            elo_l = motor_elo.obtener_rating(loc)
            elo_v = motor_elo.obtener_rating(vis)
            diff_elo.append(elo_l - elo_v)
            motor_elo.procesar_partido(loc, vis, row['Goles_L'], row['Goles_V'])

        df['Diff_ELO_Previo'] = diff_elo

        # 3. Cálculo de Promedios Reales Previos (shift(1) evita ver el futuro)
        df['Prom_Anota_L'] = df.groupby('Local')['Goles_L'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(1.3))
        df['Prom_Recibe_L'] = df.groupby('Local')['Goles_V'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(1.2))
        df['Prom_Anota_V'] = df.groupby('Visitante')['Goles_V'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(1.1))
        df['Prom_Recibe_V'] = df.groupby('Visitante')['Goles_L'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(1.3))

        self.features = ['Diff_ELO_Previo', 'Prom_Anota_L', 'Prom_Recibe_L', 'Prom_Anota_V', 'Prom_Recibe_V']
        
        if 'Corners_L' in df.columns:
            df['Prom_Corn_L'] = df.groupby('Local')['Corners_L'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(4.5))
            df['Prom_Corn_V'] = df.groupby('Visitante')['Corners_V'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(4.0))
            self.features.extend(['Prom_Corn_L', 'Prom_Corn_V'])

        if col_tarj_l in df.columns:
            df['Prom_Tarj_L'] = df.groupby('Local')[col_tarj_l].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(2.5))
            df['Prom_Tarj_V'] = df.groupby('Visitante')[col_tarj_v].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(2.5))
            self.features.extend(['Prom_Tarj_L', 'Prom_Tarj_V'])

        X = df[self.features]
        
        # 4. Entrenamiento de todos los mercados
        X_train, _, y_train, _ = train_test_split(X, df['Target_1X2'], test_size=0.1, random_state=42)
        self.model_1x2.fit(X_train, y_train)

        X_g_train, _, y_g_train, _ = train_test_split(X, df['Target_Over_Goles'], test_size=0.1, random_state=42)
        self.model_goles.fit(X_g_train, y_g_train)
        
        X_c_train, _, y_c_train, _ = train_test_split(X, df['Target_Over_Corners'], test_size=0.1, random_state=42)
        self.model_corners.fit(X_c_train, y_c_train)

        X_t_train, _, y_t_train, _ = train_test_split(X, df['Target_Over_Tarjetas'], test_size=0.1, random_state=42)
        self.model_tarjetas.fit(X_t_train, y_t_train)
        
        self.is_trained = True
        return True

    def predecir_mercados_completos(self, df_historico, equipo_local, equipo_visita, goles_sim_l, goles_sim_v, elo_local=1500, elo_visita=1500):
        if not self.is_trained:
            return {} # Respaldo vacío si falla
        
        df_l = df_historico[df_historico['Local'] == equipo_local]
        df_v = df_historico[df_historico['Visitante'] == equipo_visita]

        input_data = {
            'Diff_ELO_Previo': elo_local - elo_visita,
            'Prom_Anota_L': df_l['Goles_L'].tail(5).mean() if not df_l.empty else 1.3,
            'Prom_Recibe_L': df_l['Goles_V'].tail(5).mean() if not df_l.empty else 1.2,
            'Prom_Anota_V': df_v['Goles_V'].tail(5).mean() if not df_v.empty else 1.1,
            'Prom_Recibe_V': df_v['Goles_L'].tail(5).mean() if not df_v.empty else 1.3
        }

        if 'Prom_Corn_L' in self.features:
            input_data['Prom_Corn_L'] = df_l['Corners_L'].tail(5).mean() if not df_l.empty else 4.5
            input_data['Prom_Corn_V'] = df_v['Corners_V'].tail(5).mean() if not df_v.empty else 4.0

        col_tarj_l = 'Tarjetas_L' if 'Tarjetas_L' in df_historico.columns else 'Amarillas_L'
        col_tarj_v = 'Tarjetas_V' if 'Tarjetas_V' in df_historico.columns else 'Amarillas_V'
        
        if 'Prom_Tarj_L' in self.features:
            input_data['Prom_Tarj_L'] = df_l[col_tarj_l].tail(5).mean() if not df_l.empty else 2.5
            input_data['Prom_Tarj_V'] = df_v[col_tarj_v].tail(5).mean() if not df_v.empty else 2.5

        df_input = pd.DataFrame([input_data])[self.features]
        
        # --- PREDICCIONES ---
        probs_1x2 = self.model_1x2.predict_proba(df_input)[0]
        dict_1x2 = {c: p * 100 for c, p in zip(self.model_1x2.classes_, probs_1x2)}
        
        probs_g = self.model_goles.predict_proba(df_input)[0]
        over_g = next((p * 100 for c, p in zip(self.model_goles.classes_, probs_g) if c == 1), 0.0)

        probs_c = self.model_corners.predict_proba(df_input)[0]
        over_c = next((p * 100 for c, p in zip(self.model_corners.classes_, probs_c) if c == 1), 0.0)

        probs_t = self.model_tarjetas.predict_proba(df_input)[0]
        over_t = next((p * 100 for c, p in zip(self.model_tarjetas.classes_, probs_t) if c == 1), 0.0)

        # ====================================================================
        # ESTRUCTURA EXACTA PARA SINCRONIZAR CON MONTECARLO Y EL ESCÁNER
        # Se calculan automáticamente los Unders y se usan las llaves correctas
        # ====================================================================
        return {
            "Resultado_1X2": {
                "Gana Local": round(dict_1x2.get(1, 0.0), 1),
                "Empate": round(dict_1x2.get(0, 0.0), 1),
                "Gana Visita": round(dict_1x2.get(2, 0.0), 1)
            },
            "Goles_Over_Under": {
                "Over 2.5": round(over_g, 1),
                "Under 2.5": round(100.0 - over_g, 1)
            },
            "Corners_Totales": {
                "Over 9.5 Corners": round(over_c, 1),
                "Under 9.5 Corners": round(100.0 - over_c, 1)
            },
            "Tarjetas_Totales": {
                "Over 4.5 Tarjetas": round(over_t, 1),
                "Under 4.5 Tarjetas": round(100.0 - over_t, 1)
            }
        }
