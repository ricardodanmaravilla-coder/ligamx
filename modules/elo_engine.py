import pandas as pd

class SistemaEloLigaMX:
    def __init__(self, k_factor=20, rating_base=1500):
        # El K-factor determina qué tan rápido cambian los puntos. 20 es ideal para fútbol.
        self.k_factor = k_factor
        self.rating_base = rating_base
        self.ratings = {}

    def obtener_rating(self, equipo):
        """Devuelve el rating actual del equipo. Si no existe, le asigna 1500."""
        return self.ratings.get(equipo, self.rating_base)

    def probabilidad_esperada(self, rating_a, rating_b):
        """Calcula el porcentaje (0 a 1) de probabilidad de que A le gane a B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def procesar_partido(self, equipo_local, equipo_visita, goles_local, goles_visita):
        """Calcula y actualiza los ratings después de un resultado real."""
        rating_local = self.obtener_rating(equipo_local)
        rating_visita = self.obtener_rating(equipo_visita)

        # 1. Definir quién ganó (1 = Gana Local, 0 = Gana Visita, 0.5 = Empate)
        if goles_local > goles_visita:
            resultado_local, resultado_visita = 1, 0
        elif goles_local < goles_visita:
            resultado_local, resultado_visita = 0, 1
        else:
            resultado_local, resultado_visita = 0.5, 0.5

        # 2. Calcular qué se esperaba que pasara
        esperado_local = self.probabilidad_esperada(rating_local, rating_visita)
        esperado_visita = self.probabilidad_esperada(rating_visita, rating_local)

        # 3. Asignar los nuevos puntos
        self.ratings[equipo_local] = round(rating_local + self.k_factor * (resultado_local - esperado_local), 2)
        self.ratings[equipo_visita] = round(rating_visita + self.k_factor * (resultado_visita - esperado_visita), 2)

    def calcular_historico(self, df_partidos):
        """
        Recibe tu DataFrame histórico y lo recorre cronológicamente
        para calcular el ELO actual de toda la liga.
        """
        # Asegurarnos de que el DataFrame esté ordenado por fecha de más antiguo a más reciente
        # asumiendo que tienes una columna 'Fecha'
        df_partidos = df_partidos.sort_values(by='Fecha')

        for index, row in df_partidos.iterrows():
            self.procesar_partido(
                row['Local'], 
                row['Visita'], 
                row['Goles_Local'], 
                row['Goles_Visita']
            )
        
        # Devuelve una tabla limpia con el ranking actual
        ranking_df = pd.DataFrame(list(self.ratings.items()), columns=['Equipo', 'ELO_Rating'])
        return ranking_df.sort_values(by='ELO_Rating', ascending=False).reset_index(drop=True)
