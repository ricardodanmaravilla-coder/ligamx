import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from modules.elo_engine import SistemaEloLigaMX # Conectamos los motores

class PredictorML:
    def __init__(self):
        # max_depth=5 es vital: impide que la IA se memorice los partidos y la obliga a buscar patrones reales
        self.model_1x2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model_goles = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model_corners = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_trained = False

    def entrenar(self, df_historico):
        """Entrena la IA utilizando estrictamente datos PREVIOS a los partidos (Cero Fuga de Datos)."""
        df = df_historico.copy()
        if 'Fecha' in df.columns:
            df = df.sort_values(by='Fecha').reset_index(drop=True)
        
        # 1. Definir los Targets (Lo que queremos adivinar al final del partido)
        df['Target_1X2'] = df.apply(lambda row: 1 if row['Goles_L'] > row['Goles_V'] else (2 if row['Goles_L'] < row['Goles_V'] else 0), axis=1)
        df['Target_Over_Goles'] = ((df['Goles_L'] + df['Goles_V']) > 2.5).astype(int)
        
        if 'Corners_L' in df.columns:
            df['Target_Over_Corners'] = ((df['Corners_L'] + df['Corners_V']) > 9.5).astype(int)
        else:
            df['Target_Over_Corners'] = 0

        # 2. CONSTRUIR EL PASADO (El blindaje contra el viaje en el tiempo)
        motor_elo = SistemaEloLigaMX()
        diff_elo_lista, prom_anota_l_lista, prom_anota_v_lista = [], [], []
        hist_goles = {}
        
        for index, row in df.iterrows():
            loc = row['Local']
            vis = row['Visitante']
            
            # A. Obtener ELO actual (ANTES de que se juegue el partido)
            elo_l = motor_elo.obtener_rating(loc)
            elo_v = motor_elo.obtener_rating(vis)
            diff_elo_lista.append(elo_l - elo_v)
            
            # B. Obtener promedio de goles previo
            prom_l = hist_goles.get(loc, 1.2)
            prom_v = hist_goles.get(vis, 1.2)
            prom_anota_l_lista.append(prom_l)
            prom_anota_v_lista.append(prom_v)
            
            # C. Actualizar motores con el resultado REAL para el siguiente partido de la iteración
            motor_elo.procesar_partido(loc, vis, row['Goles_L'], row['Goles_V'])
            hist_goles[loc] = (hist_goles.get(loc, 1.2) * 4 + row['Goles_L']) / 5
            hist_goles[vis] = (hist_goles.get(vis, 1.2) * 4 + row['Goles_V']) / 5

        df['Diff_ELO_Previo'] = diff_elo_lista
        df['Prom_Goles_L_Previo'] = prom_anota_l_lista
        df['Prom_Goles_V_Previo'] = prom_anota_v_lista
        
        # 3. ENTRENAMIENTO LIMPIO: La IA solo puede ver el ELO previo y la inercia de goles
        features = ['Diff_ELO_Previo', 'Prom_Goles_L_Previo', 'Prom_Goles_V_Previo']
        X = df[features]
        
        X_train, X_test, y_train, y_test = train_test_split(X, df['Target_1X2'], test_size=0.2, random_state=42)
        self.model_1x2.fit(X_train, y_train)

        X_g_train, _, y_g_train, _ = train_test_split(X, df['Target_Over_Goles'], test_size=0.2, random_state=42)
        self.model_goles.fit(X_g_train, y_g_train)
        
        X_c_train, _, y_c_train, _ = train_test_split(X, df['Target_Over_Corners'], test_size=0.2, random_state=42)
        self.model_corners.fit(X_c_train, y_c_train)
        
        self.is_trained = True
        return True

    def predecir_mercados_completos(self, df_historico, equipo_local, equipo_visita, goles_sim_l, goles_sim_v, elo_local=1500, elo_visita=1500):
        if not self.is_trained:
            return {"1X2": {"Gana Local": 33.3, "Empate": 33.3, "Gana Visita": 33.3}, "Over_2.5_Goles": 50.0, "Over_9.5_Corners": 50.0}
        
        # Se alimenta a la IA con los datos matemáticos reales del momento actual
        df_input = pd.DataFrame([{
            'Diff_ELO_Previo': elo_local - elo_visita,
            'Prom_Goles_L_Previo': goles_sim_l, 
            'Prom_Goles_V_Previo': goles_sim_v
        }])
        
        # 1. Probabilidades 1X2
        clases_1x2 = list(self.model_1x2.classes_)
        probs_1x2 = self.model_1x2.predict_proba(df_input)[0]
        dict_1x2 = {c: p * 100 for c, p in zip(clases_1x2, probs_1x2)}
        
        resultado_1x2 = {
            "Gana Local": round(dict_1x2.get(1, 0.0), 1),
            "Empate": round(dict_1x2.get(0, 0.0), 1),
            "Gana Visita": round(dict_1x2.get(2, 0.0), 1)
        }

        # 2. Goles y Corners
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
