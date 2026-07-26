import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from modules.elo_engine import SistemaEloLigaMX

class PredictorML:
    def __init__(self):
        # Configuración estricta para evitar sobreajuste y probabilidades irreales
        self.model_1x2 = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15, random_state=42)
        self.model_goles = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15, random_state=42)
        self.model_corners = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15, random_state=42)
        self.is_trained = False
        self.features = []

    def entrenar(self, df_historico):
        """El ML escanea el CSV de forma independiente y aprende de todas las variables."""
        df = df_historico.copy()
        if 'Fecha' in df.columns:
            df = df.sort_values(by='Fecha').reset_index(drop=True)
        
        # 1. Definir los resultados reales (Targets)
        df['Target_1X2'] = df.apply(lambda row: 1 if row['Goles_L'] > row['Goles_V'] else (2 if row['Goles_L'] < row['Goles_V'] else 0), axis=1)
        df['Target_Over_Goles'] = ((df['Goles_L'] + df['Goles_V']) > 2.5).astype(int)
        df['Target_Over_Corners'] = ((df['Corners_L'] + df['Corners_V']) > 9.5).astype(int) if 'Corners_L' in df.columns else 0

        # 2. Análisis Dinámico de la Liga (Evitando Fuga de Datos con shift(1))
        # Escaneamos poder ofensivo y defensivo de los últimos 5 partidos
        df['Prom_Anota_L'] = df.groupby('Local')['Goles_L'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(1.3))
        df['Prom_Recibe_L'] = df.groupby('Local')['Goles_V'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(1.2))
        
        df['Prom_Anota_V'] = df.groupby('Visitante')['Goles_V'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(1.1))
        df['Prom_Recibe_V'] = df.groupby('Visitante')['Goles_L'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(1.3))

        # Reconstrucción del ELO
        motor_elo = SistemaEloLigaMX()
        diff_elo = []
        for index, row in df.iterrows():
            loc, vis = row['Local'], row['Visitante']
            elo_l = motor_elo.obtener_rating(loc)
            elo_v = motor_elo.obtener_rating(vis)
            diff_elo.append(elo_l - elo_v)
            motor_elo.procesar_partido(loc, vis, row['Goles_L'], row['Goles_V'])

        df['Diff_ELO_Previo'] = diff_elo

        # Definir las variables maestras que aprenderá el modelo
        self.features = ['Diff_ELO_Previo', 'Prom_Anota_L', 'Prom_Recibe_L', 'Prom_Anota_V', 'Prom_Recibe_V']
        
        # Si tienes Corners en el CSV, los añade a su inteligencia
        if 'Corners_L' in df.columns:
            df['Prom_Corn_L'] = df.groupby('Local')['Corners_L'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(4.5))
            df['Prom_Corn_V'] = df.groupby('Visitante')['Corners_V'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(4.0))
            self.features.extend(['Prom_Corn_L', 'Prom_Corn_V'])

        X = df[self.features]
        
        # 3. Entrenamiento Limpio
        X_train, _, y_train, _ = train_test_split(X, df['Target_1X2'], test_size=0.1, random_state=42)
        self.model_1x2.fit(X_train, y_train)

        X_g_train, _, y_g_train, _ = train_test_split(X, df['Target_Over_Goles'], test_size=0.1, random_state=42)
        self.model_goles.fit(X_g_train, y_g_train)
        
        X_c_train, _, y_c_train, _ = train_test_split(X, df['Target_Over_Corners'], test_size=0.1, random_state=42)
        self.model_corners.fit(X_c_train, y_c_train)
        
        self.is_trained = True
        return True

    def predecir_mercados_completos(self, df_historico, equipo_local, equipo_visita, goles_sim_l, goles_sim_v, elo_local=1500, elo_visita=1500):
        """
        Predicción 100% independiente.
        NOTA: 'goles_sim_l' y 'goles_sim_v' se reciben para no romper el escáner, pero se IGNORAN.
        """
        if not self.is_trained:
            return {"1X2": {"Gana Local": 33.3, "Empate": 33.3, "Gana Visita": 33.3}, "Over_2.5_Goles": 50.0, "Over_9.5_Corners": 50.0}
        
        # 1. El modelo escanea el CSV por su cuenta para los equipos actuales
        df_l = df_historico[df_historico['Local'] == equipo_local]
        df_v = df_historico[df_historico['Visitante'] == equipo_visita]

        # Extraemos el poder real reciente
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

        df_input = pd.DataFrame([input_data])[self.features]
        
        # 2. Predicciones Independientes
        clases_1x2 = list(self.model_1x2.classes_)
        probs_1x2 = self.model_1x2.predict_proba(df_input)[0]
        dict_1x2 = {c: p * 100 for c, p in zip(clases_1x2, probs_1x2)}
        
        resultado_1x2 = {
            "Gana Local": round(dict_1x2.get(1, 0.0), 1),
            "Empate": round(dict_1x2.get(0, 0.0), 1),
            "Gana Visita": round(dict_1x2.get(2, 0.0), 1)
        }

        probs_goles = self.model_goles.predict_proba(df_input)[0]
        clases_goles = list(self.model_goles.classes_)
        over_goles_prob = next((p * 100 for c, p in zip(clases_goles, probs_goles) if c == 1), 0.0)

        probs_corners = self.model_corners.predict_proba(df_input)[0]
        clases_corners = list(self.model_corners.classes_)
        over_corners_prob = next((p * 100 for c, p in zip(clases_corners, probs_corners) if c == 1), 0.0)

        return {
            "1X2": resultado_1x2,
            "Over_2.5_Goles": round(over_goles_prob, 1),
            "Over_9.5_Corners": round(over_corners_prob, 1)
        }
