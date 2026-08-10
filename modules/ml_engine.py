import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class PredictorML:
    def __init__(self):
        self.model_1x2 = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_goles = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_corners = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_tarjetas = RandomForestRegressor(n_estimators=100, random_state=42)
        self.encoder_equipos = LabelEncoder()
        self.is_trained = False

    def entrenar(self, df_historico):
        try:
            if df_historico is None or len(df_historico) < 20:
                return False
                
            df = df_historico.copy()
            df['Local'] = df['Local'].astype(str).str.strip()
            df['Visitante'] = df['Visitante'].astype(str).str.strip()
            
            todos_equipos = pd.concat([df['Local'], df['Visitante']]).unique()
            self.encoder_equipos.fit(todos_equipos)
            
            df['Local_Encoded'] = self.encoder_equipos.transform(df['Local'])
            df['Visita_Encoded'] = self.encoder_equipos.transform(df['Visitante'])
            
            def resultado_partido(row):
                gl = row.get('Goles_L', row.get('Goles_Local', 0))
                gv = row.get('Goles_V', row.get('Goles_Visita', 0))
                if gl > gv: return 2
                elif gl < gv: return 0
                else: return 1
                
            df['Target_1X2'] = df.apply(resultado_partido, axis=1)
            df['Total_Goles'] = df.get('Goles_L', df.get('Goles_Local', 0)) + df.get('Goles_V', df.get('Goles_Visita', 0))
            
            c_l = df.get('Corners_L', 5.0)
            c_v = df.get('Corners_V', 4.5)
            df['Total_Corners'] = c_l.fillna(5.0) + c_v.fillna(4.5)

            am_l = df.get('Amarillas_L', 2.0).fillna(2.0)
            rj_l = df.get('Rojas_L', 0.0).fillna(0.0)
            am_v = df.get('Amarillas_V', 2.2).fillna(2.2)
            rj_v = df.get('Rojas_V', 0.0).fillna(0.0)
            df['Total_Tarjetas'] = (am_l + rj_l * 2) + (am_v + rj_v * 2)

            features = ['Local_Encoded', 'Visita_Encoded']
            if 'ELO_Local' in df.columns and 'ELO_Visita' in df.columns:
                df['Diff_ELO'] = df['ELO_Local'] - df['ELO_Visita']
                features.append('Diff_ELO')
            else:
                df['Diff_ELO'] = 0.0
                features.append('Diff_ELO')

            X = df[features].fillna(0)
            
            self.model_1x2.fit(X, df['Target_1X2'])
            self.model_goles.fit(X, df['Total_Goles'])
            self.model_corners.fit(X, df['Total_Corners'])
            self.model_tarjetas.fit(X, df['Total_Tarjetas'])
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error entrenando ML: {e}")
            return False

    # ... (mantén las importaciones y método __init__ y entrenar igual)

    def predecir_mercados_completos(self, df_historico, equipo_local, equipo_visita, goles_sim_l, goles_sim_v, elo_local=1500, elo_visita=1500, linea_goles=2.5, linea_corners=9.5, linea_tarjetas=4.5):
        if not self.is_trained:
            if not self.entrenar(df_historico):
                return {}
                
        try:
            known_classes = set(self.encoder_equipos.classes_)
            loc_enc = self.encoder_equipos.transform([equipo_local])[0] if equipo_local in known_classes else 0
            vis_enc = self.encoder_equipos.transform([equipo_visita])[0] if equipo_visita in known_classes else 0
            
            diff_elo = elo_local - elo_visita
            X_pred = pd.DataFrame([[loc_enc, vis_enc, diff_elo]], columns=['Local_Encoded', 'Visita_Encoded', 'Diff_ELO'])
            
            probs_1x2 = self.model_1x2.predict_proba(X_pred)[0]
            classes = self.model_1x2.classes_
            
            p_visita, p_empate, p_local = 33.3, 33.3, 33.4
            for cls, prob in zip(classes, probs_1x2):
                if cls == 0: p_visita = round(float(prob) * 100, 1)
                elif cls == 1: p_empate = round(float(prob) * 100, 1)
                elif cls == 2: p_local = round(float(prob) * 100, 1)

            goles_totales_ml = float(self.model_goles.predict(X_pred)[0])
            corners_totales_ml = float(self.model_corners.predict(X_pred)[0])
            tarjetas_totales_ml = float(self.model_tarjetas.predict(X_pred)[0])
            
            over_goles_ml = round(min(95.0, max(5.0, (goles_totales_ml / (linea_goles + 0.3)) * 55.0)), 1)
            under_goles_ml = round(100.0 - over_goles_ml, 1)

            over_corners_ml = round(min(95.0, max(5.0, (corners_totales_ml / (linea_corners + 0.5)) * 50.0)), 1)
            under_corners_ml = round(100.0 - over_corners_ml, 1)

            over_tarjetas_ml = round(min(95.0, max(5.0, (tarjetas_totales_ml / (linea_tarjetas + 0.5)) * 50.0)), 1)
            under_tarjetas_ml = round(100.0 - over_tarjetas_ml, 1)
            
            return {
                "Resultado_1X2": {
                    "Gana Local": p_local,
                    "Empate": p_empate,
                    "Gana Visita": p_visita
                },
                "Goles_Over_Under": {
                    f"Over {linea_goles}": over_goles_ml,
                    f"Under {linea_goles}": under_goles_ml
                },
                "Corners_Totales": {
                    f"Over {linea_corners} Corners": over_corners_ml,
                    f"Under {linea_corners} Corners": under_corners_ml
                },
                "Tarjetas_Totales": {
                    f"Over {linea_tarjetas} Tarjetas": over_tarjetas_ml,
                    f"Under {linea_tarjetas} Tarjetas": under_tarjetas_ml
                }
            }
        except Exception as e:
            print(f"Error en predicción ML: {e}")
            return {}
