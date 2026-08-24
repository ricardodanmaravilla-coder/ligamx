import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from .feature_engineering import add_rolling_features, current_match_features, normalize_team

class PredictorML:
    def __init__(self, random_state=42):
        self.model_1x2=RandomForestClassifier(n_estimators=350,min_samples_leaf=8,max_features='sqrt',class_weight='balanced_subsample',random_state=random_state,n_jobs=-1)
        self.reg_goles=RandomForestRegressor(n_estimators=350,min_samples_leaf=8,max_features=.7,random_state=random_state,n_jobs=-1)
        self.reg_corners=RandomForestRegressor(n_estimators=350,min_samples_leaf=8,max_features=.7,random_state=random_state,n_jobs=-1)
        self.reg_cards=RandomForestRegressor(n_estimators=350,min_samples_leaf=8,max_features=.7,random_state=random_state,n_jobs=-1)
        self.features=[]; self.resid_g=np.array([]); self.resid_c=np.array([]); self.resid_t=np.array([]); self.is_trained=False

    def _prep(self, df):
        d=add_rolling_features(df)
        prefixes=('H_','A_')
        self.features=['Diff_ELO_Pre']+[c for c in d.columns if c.startswith(prefixes) and not c.startswith(('H_idx','A_idx'))]
        self.features=[c for c in self.features if pd.api.types.is_numeric_dtype(d[c])]
        d['Target_1X2']=np.where(d.Goles_L>d.Goles_V,2,np.where(d.Goles_L<d.Goles_V,0,1))
        d['Total_Goles']=d.Goles_L+d.Goles_V; d['Total_Corners']=d.Corners_L+d.Corners_V
        d['Total_Tarjetas']=d.Amarillas_L+2*d.Rojas_L+d.Amarillas_V+2*d.Rojas_V
        d=d.dropna(subset=self.features+['Target_1X2','Total_Goles','Total_Corners','Total_Tarjetas']).reset_index(drop=True)
        return d

    def entrenar(self, df_historico):
        d=self._prep(df_historico)
        if len(d)<300: return False
        split=max(int(len(d)*.8),len(d)-300)
        tr,cal=d.iloc[:split],d.iloc[split:]
        X=tr[self.features]
        self.model_1x2.fit(X,tr.Target_1X2); self.reg_goles.fit(X,tr.Total_Goles); self.reg_corners.fit(X,tr.Total_Corners); self.reg_cards.fit(X,tr.Total_Tarjetas)
        Xc=cal[self.features]
        self.resid_g=(cal.Total_Goles-self.reg_goles.predict(Xc)).to_numpy()
        self.resid_c=(cal.Total_Corners-self.reg_corners.predict(Xc)).to_numpy()
        self.resid_t=(cal.Total_Tarjetas-self.reg_cards.predict(Xc)).to_numpy()
        self.is_trained=True
        return True

    @staticmethod
    def _over_prob(pred,line,resid):
        if resid.size<50: return np.nan
        return float(np.mean(pred+resid>float(line))*100)

    def predecir_mercados_completos(self, df_historico, equipo_local, equipo_visita, goles_sim_l=None, goles_sim_v=None, elo_local=None, elo_visita=None, linea_goles=2.5, linea_corners=9.5, linea_tarjetas=4.5):
        if not self.is_trained and not self.entrenar(df_historico): return {}
        f=current_match_features(df_historico,normalize_team(equipo_local),normalize_team(equipo_visita))
        X=pd.DataFrame([{c:f.get(c,np.nan) for c in self.features}])
        if X.isna().any(axis=None): raise ValueError('Features prepartido incompletas: NO BET')
        probs=dict(zip(self.model_1x2.classes_,self.model_1x2.predict_proba(X)[0]))
        pg=float(self.reg_goles.predict(X)[0]); pc=float(self.reg_corners.predict(X)[0]); pt=float(self.reg_cards.predict(X)[0])
        og=self._over_prob(pg,linea_goles,self.resid_g); oc=self._over_prob(pc,linea_corners,self.resid_c); ot=self._over_prob(pt,linea_tarjetas,self.resid_t)
        return {'Resultado_1X2':{'Gana Local':round(probs.get(2,0)*100,1),'Empate':round(probs.get(1,0)*100,1),'Gana Visita':round(probs.get(0,0)*100,1)},
                'Goles_Over_Under':{f'Over {linea_goles}':round(og,1),f'Under {linea_goles}':round(100-og,1)},
                'Corners_Totales':{f'Over {linea_corners} Corners':round(oc,1),f'Under {linea_corners} Corners':round(100-oc,1)},
                'Tarjetas_Totales':{f'Over {linea_tarjetas} Tarjetas':round(ot,1),f'Under {linea_tarjetas} Tarjetas':round(100-ot,1)}}
