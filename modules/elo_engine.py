import pandas as pd
import numpy as np

class SistemaEloLigaMX:
    def __init__(self, k_factor=32, base_rating=1500):
        self.k_factor = k_factor
        self.base_rating = base_rating

    def calcular_historico(self, df_historico):
        """
        Calcula el ELO actual y evolutivo de cada equipo partido a partido 
        basándose en el archivo histórico real.
        """
        if df_historico is None or df_historico.empty:
            return pd.DataFrame(columns=['Equipo', 'ELO_Rating', 'Partidos_Jugados'])

        df = df_historico.copy()
        
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            df = df.sort_values('Fecha').reset_index(drop=True)

        ratings = {}
        partidos_jugados = {}

        def get_rating(eq):
            if eq not in ratings:
                ratings[eq] = self.base_rating
                partidos_jugados[eq] = 0
            return ratings[eq]

        for idx, row in df.iterrows():
            local = str(row['Local']).strip()
            visita = str(row['Visitante']).strip()
            
            gl = row.get('Goles_L', row.get('Goles_Local', 0))
            gv = row.get('Goles_V', row.get('Goles_Visita', 0))
            
            if pd.isna(gl) or pd.isna(gv):
                continue

            elo_l = get_rating(local)
            elo_v = get_rating(visita)

            elo_l_adj = elo_l + 35

            exp_l = 1 / (1 + 10 ** ((elo_v - elo_l_adj) / 400))
            exp_v = 1 / (1 + 10 ** ((elo_l_adj - elo_v) / 400))

            if gl > gv:
                s_l, s_v = 1.0, 0.0
            elif gl < gv:
                s_l, s_v = 0.0, 1.0
            else:
                s_l, s_v = 0.5, 0.5

            ratings[local] = elo_l + self.k_factor * (s_l - exp_l)
            ratings[visita] = elo_v + self.k_factor * (s_v - exp_v)

            partidos_jugados[local] = partidos_jugados.get(local, 0) + 1
            partidos_jugados[visita] = partidos_jugados.get(visita, 0) + 1

        data_ranking = []
        for eq, rating in ratings.items():
            data_ranking.append({
                'Equipo': eq,
                'ELO_Rating': round(float(rating), 1),
                'Partidos_Jugados': partidos_jugados.get(eq, 0)
            })

        df_ranking = pd.DataFrame(data_ranking)
        df_ranking = df_ranking.sort_values(by='ELO_Rating', ascending=False).reset_index(drop=True)
        return df_ranking
