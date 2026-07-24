import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class PredictorML:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def preparar_datos(self, df_historico):
        """Prepara las variables y etiquetas utilizando exclusivamente datos reales del histórico."""
        df = df_historico.copy()
        
        # Etiqueta real: 1 si el local ganó el partido, 0 en otro caso
        df['Target_Gana_Local'] = (df['Goles_L'] > df['Goles_V']).astype(int)
        
        # Features basados en las columnas reales de tu base de datos
        features = ['Goles_L', 'Goles_V', 'xG_L', 'xG_V', 'Corners_L', 'Corners_V', 'Amarillas_L', 'Amarillas_V']
        
        df = df.dropna(subset=features + ['Target_Gana_Local'])
        
        X = df[features]
        y = df['Target_Gana_Local']
        return X, y

    def entrenar(self, df_historico):
        """Entrena el modelo de Machine Learning con el historial real."""
        try:
            X, y = self.preparar_datos(df_historico)
            if len(X) < 10:
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error entrenando ML con datos reales: {e}")
            return False

    def obtener_promedios_reales(self, df_historico, equipo_local, equipo_visita):
        """Calcula los promedios reales de xG, corners y tarjetas de los últimos partidos de cada equipo."""
        # Promedios históricos reales del equipo local jugando en casa o general
        loc_data = df_historico[df_historico['Local'] == equipo_local]
        vis_data = df_historico[df_historico['Visitante'] == equipo_visita]
        
        x_g_l = loc_data['xG_L'].mean() if not loc_data.empty and 'xG_L' in loc_data else 1.2
        x_g_v = vis_data['xG_V'].mean() if not vis_data.empty and 'xG_V' in vis_data else 1.0
        
        corners_l = loc_data['Corners_L'].mean() if not loc_data.empty and 'Corners_L' in loc_data else 4.5
        corners_v = vis_data['Corners_V'].mean() if not vis_data.empty and 'Corners_V' in vis_data else 4.0
        
        amarillas_l = loc_data['Amarillas_L'].mean() if not loc_data.empty and 'Amarillas_L' in loc_data else 2.0
        amarillas_v = vis_data['Amarillas_V'].mean() if not vis_data.empty and 'Amarillas_V' in vis_data else 2.0

        return {
            'xG_L': float(x_g_l),
            'xG_V': float(x_g_v),
            'Corners_L': float(corners_l),
            'Corners_V': float(corners_v),
            'Amarillas_L': float(amarillas_l),
            'Amarillas_V': float(amarillas_v)
        }

    def predecir_partido_real(self, df_historico, equipo_local, equipo_visita, goles_estimados_l, goles_estimados_v):
        """Alimenta al modelo con los goles de Montecarlo y las estadísticas reales extraídas del CSV."""
        if not self.is_trained:
            return 0.5
        
        promedios = self.obtener_promedios_reales(df_historico, equipo_local, equipo_visita)
        
        stats_reales = {
            'Goles_L': goles_estimados_l,
            'Goles_V': goles_estimados_v,
            'xG_L': promedios['xG_L'],
            'xG_V': promedios['xG_V'],
            'Corners_L': promedios['Corners_L'],
            'Corners_V': promedios['Corners_V'],
            'Amarillas_L': promedios['Amarillas_L'],
            'Amarillas_V': promedios['Amarillas_V']
        }
        
        features = ['Goles_L', 'Goles_V', 'xG_L', 'xG_V', 'Corners_L', 'Corners_V', 'Amarillas_L', 'Amarillas_V']
        df_input = pd.DataFrame([stats_reales])[features]
        
        probabilidades = self.model.predict_proba(df_input)
        return float(probabilidades[0][1])
