import pandas as pd

class SistemaEloLigaMX:
    def __init__(self, k_factor=35, rating_base=1500, home_advantage=75):
        # k_factor ajustado a 35 para reflejar rachas en torneos cortos
        self.k_factor = k_factor
        self.rating_base = rating_base
        # 75 puntos extra de ELO artificial al local para reflejar la dificultad de visita
        self.home_advantage = home_advantage 
        self.ratings = {}

    def obtener_rating(self, equipo):
        """Devuelve el rating actual del equipo. Si no existe, le asigna 1500."""
        return self.ratings.get(equipo, self.rating_base)

    def probabilidad_esperada(self, rating_a, rating_b):
        """Calcula el porcentaje (0 a 1) de probabilidad de que A le gane a B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def procesar_partido(self, equipo_local, equipo_visita, goles_local, goles_visita):
        """Calcula y actualiza los ratings después de un resultado real, usando HFA y MoV."""
        rating_local = self.obtener_rating(equipo_local)
        rating_visita = self.obtener_rating(equipo_visita)

        # 1. Aplicar la Ventaja de Localía (HFA) SOLO para calcular la expectativa
        # Esto asume que al local se le exige ganar por estar en casa.
        rating_local_ajustado = rating_local + self.home_advantage

        # 2. Definir quién ganó (1 = Gana Local, 0 = Gana Visita, 0.5 = Empate)
        if goles_local > goles_visita:
            resultado_local, resultado_visita = 1, 0
        elif goles_local < goles_visita:
            resultado_local, resultado_visita = 0, 1
        else:
            resultado_local, resultado_visita = 0.5, 0.5

        # 3. Calcular qué se esperaba que pasara usando el rating ajustado
        esperado_local = self.probabilidad_esperada(rating_local_ajustado, rating_visita)
        esperado_visita = self.probabilidad_esperada(rating_visita, rating_local_ajustado)

        # 4. Multiplicador por Margen de Victoria (MoV)
        diff = abs(goles_local - goles_visita)
        if diff <= 1:
            mov = 1.0
        elif diff == 2:
            mov = 1.5
        else:
            # Fórmula estándar de fútbol para goleadas (3 o más goles)
            mov = (11.0 + diff) / 8.0

        # 5. Asignar los nuevos puntos reales (aplicando el multiplicador MoV)
        cambio_local = self.k_factor * mov * (resultado_local - esperado_local)
        cambio_visita = self.k_factor * mov * (resultado_visita - esperado_visita)

        self.ratings[equipo_local] = round(rating_local + cambio_local, 2)
        self.ratings[equipo_visita] = round(rating_visita + cambio_visita, 2)

    def calcular_historico(self, df_partidos):
        """
        Recorre cronológicamente el DataFrame para calcular el ELO actual.
        """
        if 'Fecha' in df_partidos.columns:
            df_partidos = df_partidos.sort_values(by='Fecha')

        for index, row in df_partidos.iterrows():
            self.procesar_partido(
                row['Local'],         
                row['Visitante'],        
                row['Goles_L'],   
                row['Goles_V']   
            )
        
        # Devuelve una tabla limpia con el ranking actual
        ranking_df = pd.DataFrame(list(self.ratings.items()), columns=['Equipo', 'ELO_Rating'])
        return ranking_df.sort_values(by='ELO_Rating', ascending=False).reset_index(drop=True)
