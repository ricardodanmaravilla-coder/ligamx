import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class PredictorML:
    def __init__(self):
        # Creamos tres modelos independientes: 1 para el 1X2, 1 para Goles y 1 para Corners
        self.model_1x2 = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_goles = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_corners = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def entrenar(self, df_historico):
        """Entrena los modelos de Machine Learning utilizando el historial real."""
        try:
            df = df_historico.copy()
            
            # --- 1. PREPARAR DATOS PARA 1X2 ---
            def obtener_resultado_1x2(row):
                if row['Goles_L'] > row['Goles_V']: return 1 # Gana Local
                elif row['Goles_L'] < row['Goles_V']: return 2 # Gana Visitante
                else: return 0 # Empate

            df['Target_1X2'] = df.apply(obtener_resultado_1x2, axis=1)
            
            # --- 2. PREPARAR DATOS PARA GOLES (Over 2.5 = 1, Under = 0) ---
            df['Total_Goles'] = df['Goles_L'] + df['Goles_V']
            df['Target_Over_Goles'] = (df['Total_Goles'] > 2.5).astype(int)

            # --- 3. PREPARAR DATOS PARA CORNERS (Over 9.5 = 1, Under = 0) ---
            if 'Corners_L' in df.columns and 'Corners_V' in df.columns:
                df['Total_Corners'] = df['Corners_L'] + df['Corners_V']
                df['Target_Over_Corners'] = (df['Total_Corners'] > 9.5).astype(int)
            else:
                df['Target_Over_Corners'] = 0

            # --- 4. PREPARAR DATOS DE FUERZA RELATIVA (ELO) ---
            # Si el histórico tiene puntuación ELO pre-calculada, el modelo la usará
            if 'ELO_Local' in df.columns and 'ELO_Visita' in df.columns:
                df['Diff_ELO'] = df['ELO_Local'] - df['ELO_Visita']
            else:
                df['Diff_ELO'] = 0.0 # Respaldo seguro si no existen las columnas

            # Integramos Diff_ELO a las características (features) que el modelo va a aprender
            features = ['Goles_L', 'Goles_V', 'xG_L', 'xG_V', 'Corners_L', 'Corners_V', 'Amarillas_L', 'Amarillas_V', 'Diff_ELO']
            df = df.dropna(subset=features + ['Target_1X2', 'Target_Over_Goles'])

            X = df[features]
            
            # Entrenamos Modelo 1X2
            X_train, X_test, y_train, y_test = train_test_split(X, df['Target_1X2'], test_size=0.2, random_state=42)
            self.model_1x2.fit(X_train, y_train)

            # Entrenamos Modelo Goles Over/Under 2.5
            X_g_train, _, y_g_train, _ = train_test_split(X, df['Target_Over_Goles'], test_size=0.2, random_state=42)
            self.model_goles.fit(X_g_train, y_g_train)

            # Entrenamos Modelo Corners Over/Under 9.5
            X_c_train, _, y_c_train, _ = train_test_split(X, df['Target_Over_Corners'], test_size=0.2, random_state=42)
            self.model_corners.fit(X_c_train, y_c_train)

            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error entrenando modelos ML multiclase: {e}")
            return False

    def obtener_promedios_reales(self, df_historico, equipo_local, equipo_visita):
        """Calcula los promedios reales de xG, corners y tarjetas."""
        loc_data = df_historico[df_historico['Local'] == equipo_local]
        vis_data = df_historico[df_historico['Visitante'] == equipo_visita]
        
        return {
            'xG_L': float(loc_data['xG_L'].mean() if not loc_data.empty and 'xG_L' in loc_data else 1.2),
            'xG_V': float(vis_data['xG_V'].mean() if not vis_data.empty and 'xG_V' in vis_data else 1.0),
            'Corners_L': float(loc_data['Corners_L'].mean() if not loc_data.empty and 'Corners_L' in loc_data else 4.5),
            'Corners_V': float(vis_data['Corners_V'].mean() if not vis_data.empty and 'Corners_V' in vis_data else 4.0),
            'Amarillas_L': float(loc_data['Amarillas_L'].mean() if not loc_data.empty and 'Amarillas_L' in loc_data else 2.0),
            'Amarillas_V': float(vis_data['Amarillas_V'].mean() if not vis_data.empty and 'Amarillas_V' in vis_data else 2.0)
        }

    # SE AÑADEN elo_local Y elo_visita COMO PARÁMETROS OPCIONALES
    def predecir_mercados_completos(self, df_historico, equipo_local, equipo_visita, goles_sim_l, goles_sim_v, elo_local=1500, elo_visita=1500):
        """Predice de forma independiente 1X2, Goles (Over 2.5) y Corners (Over 9.5) mediante Machine Learning."""
        if not self.is_trained:
            return {"1X2": {"Gana Local": 33.3, "Empate": 33.3, "Gana Visita": 33.3}, "Over_2.5_Goles": 50.0, "Over_9.5_Corners": 50.0}
        
        promedios = self.obtener_promedios_reales(df_historico, equipo_local, equipo_visita)
        
        stats_reales = {
            'Goles_L': goles_sim_l,
            'Goles_V': goles_sim_v,
            'xG_L': promedios['xG_L'],
            'xG_V': promedios['xG_V'],
            'Corners_L': promedios['Corners_L'],
            'Corners_V': promedios['Corners_V'],
            'Amarillas_L': promedios['Amarillas_L'],
            'Amarillas_V': promedios['Amarillas_V'],
            'Diff_ELO': elo_local - elo_visita # Inyectamos la fuerza actual al modelo
        }
        
        features = ['Goles_L', 'Goles_V', 'xG_L', 'xG_V', 'Corners_L', 'Corners_V', 'Amarillas_L', 'Amarillas_V', 'Diff_ELO']
        df_input = pd.DataFrame([stats_reales])[features]
        
        # 1. Probabilidades 1X2
        clases_1x2 = list(self.model_1x2.classes_)
        probs_1x2 = self.model_1x2.predict_proba(df_input)[0]
        dict_1x2 = {c: p * 100 for c, p in zip(clases_1x2, probs_1x2)}
        
        resultado_1x2 = {
            "Gana Local": round(dict_1x2.get(1, 0.0), 1),
            "Empate": round(dict_1x2.get(0, 0.0), 1),
            "Gana Visita": round(dict_1x2.get(2, 0.0), 1)
        }

        # 2. Probabilidad Over 2.5 Goles
        probs_goles = self.model_goles.predict_proba(df_input)[0]
        # Si la clase 1 representa el Over
        clases_goles = list(self.model_goles.classes_)
        over_goles_prob = 0.0
        for c, p in zip(clases_goles, probs_goles):
            if c == 1: over_goles_prob = p * 100

        # 3. Probabilidad Over 9.5 Corners
        probs_corners = self.model_corners.predict_proba(df_input)[0]
        clases_corners = list(self.model_corners.classes_)
        over_corners_prob = 0.0
        for c, p in zip(clases_corners, probs_corners):
            if c == 1: over_corners_prob = p * 100

        return {
            "1X2": resultado_1x2,
            "Over_2.5_Goles": round(over_goles_prob, 1),
            "Over_9.5_Corners": round(over_corners_prob, 1)
        }
