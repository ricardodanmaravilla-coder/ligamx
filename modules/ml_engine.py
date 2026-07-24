import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class PredictorML:
    def __init__(self):
        # Usamos multi_class implícito en RandomForest para clasificar 0 (Empate), 1 (Gana Local) y 2 (Gana Visita)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def preparar_datos(self, df_historico):
        """Prepara las variables y etiquetas 1X2 utilizando exclusivamente datos reales del histórico."""
        df = df_historico.copy()
        
        # Definimos el resultado real (Target):
        # 1 = Gana Local, 2 = Gana Visitante, 0 = Empate
        def obtener_resultado_1x2(row):
            if row['Goles_L'] > row['Goles_V']:
                return 1 # Gana Local
            elif row['Goles_L'] < row['Goles_V']:
                return 2 # Gana Visitante
            else:
                return 0 # Empate

        df['Target_1X2'] = df.apply(obtener_resultado_1x2, axis=1)
        
        features = ['Goles_L', 'Goles_V', 'xG_L', 'xG_V', 'Corners_L', 'Corners_V', 'Amarillas_L', 'Amarillas_V']
        df = df.dropna(subset=features + ['Target_1X2'])
        
        X = df[features]
        y = df['Target_1X2']
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
            print(f"Error entrenando ML: {e}")
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

    def predecir_partido_real_1x2(self, df_historico, equipo_local, equipo_visita, goles_estimados_l, goles_estimados_v):
        """Devuelve un diccionario con las probabilidades porcentuales para Gana Local, Empate y Gana Visita."""
        if not self.is_trained:
            return {"Gana Local": 33.3, "Empate": 33.3, "Gana Visita": 33.3}
        
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
        
        # predict_proba devuelve el orden de las clases ordenadas: [0 (Empate), 1 (Local), 2 (Visita)] dependiendo de cómo las encuadre sklearn
        # Mapeamos las clases de forma segura:
        clases = list(self.model.classes_)
        probabilidades = self.model.predict_proba(df_input)[0]
        
        prob_dict = {clase: prob * 100 for clase, prob in zip(clases, probabilidades)}
        
        return {
            "Gana Local": round(prob_dict.get(1, 0.0), 1),
            "Empate": round(prob_dict.get(0, 0.0), 1),
            "Gana Visita": round(prob_dict.get(2, 0.0), 1)
        }
