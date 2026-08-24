import pandas as pd
from .feature_engineering import add_pre_match_elo, clean_history, normalize_team

class SistemaEloLigaMX:
    def __init__(self, k_factor=24, base_rating=1500, home_advantage=45):
        self.k_factor=k_factor; self.base_rating=base_rating; self.home_advantage=home_advantage

    def calcular_historico(self, df_historico):
        df=add_pre_match_elo(df_historico, self.k_factor, self.home_advantage, self.base_rating)
        ratings={}; games={}
        for _,r in df.iterrows():
            h,a=r.Local,r.Visitante
            rh,ra=ratings.get(h,self.base_rating),ratings.get(a,self.base_rating)
            exp=1/(1+10**((ra-(rh+self.home_advantage))/400))
            s=1 if r.Goles_L>r.Goles_V else 0 if r.Goles_L<r.Goles_V else .5
            margin=abs(float(r.Goles_L)-float(r.Goles_V))
            import numpy as np
            delta=self.k_factor*(1+0.15*np.log1p(margin))*(s-exp)
            ratings[h]=rh+delta; ratings[a]=ra-delta
            games[h]=games.get(h,0)+1; games[a]=games.get(a,0)+1
        return pd.DataFrame([{'Equipo':k,'ELO_Rating':round(v,1),'Partidos_Jugados':games[k]} for k,v in ratings.items()]).sort_values('ELO_Rating',ascending=False).reset_index(drop=True)

    def elo_prepartido(self, df_historico, local, visita):
        local,visita=normalize_team(local),normalize_team(visita)
        table=self.calcular_historico(df_historico)
        d=dict(zip(table.Equipo,table.ELO_Rating))
        if local not in d or visita not in d: raise ValueError('Equipo sin ELO suficiente')
        return float(d[local]), float(d[visita])
