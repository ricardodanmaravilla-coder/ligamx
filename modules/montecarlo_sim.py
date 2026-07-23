import pandas as pd
import numpy as np

def aplicar_dixon_coles(lambda_l, lambda_v, prob_matriz, rho=-0.15):
    """
    Ajusta la matriz de probabilidades de marcadores exactos utilizando 
    la correlación de Dixon-Coles para inflar los empates de baja anotación.
    """
    # Evitar resultados negativos en tau limitando el impacto de rho
    if (1 - lambda_l * lambda_v * rho) < 0:
        rho = 0
        
    tau_0_0 = 1 - lambda_l * lambda_v * rho
    tau_1_0 = 1 + lambda_v * rho
    tau_0_1 = 1 + lambda_l * rho
    tau_1_1 = 1 - rho

    # prob_matriz es un dict con llave "x-y" y valor probabilistico
    if "0-0" in prob_matriz: prob_matriz["0-0"] *= tau_0_0
    if "1-0" in prob_matriz: prob_matriz["1-0"] *= tau_1_0
    if "0-1" in prob_matriz: prob_matriz["0-1"] *= tau_0_1
    if "1-1" in prob_matriz: prob_matriz["1-1"] *= tau_1_1

    # Normalizar la matriz para que vuelva a sumar 100%
    total = sum(prob_matriz.values())
    for k in prob_matriz.keys():
        prob_matriz[k] = (prob_matriz[k] / total)

    return prob_matriz

def calcular_lambdas_ajustadas(df, equipo_local, equipo_visita):
    """
    Calcula la expectativa de goles (Lambda) ajustada por la fuerza de ataque 
    y defensa relativa al promedio de toda la Liga MX.
    """
    # 1. Calcular promedios globales de la liga
    # (Asumiendo que las columnas se llaman 'Goles_L' y 'Goles_V' o similar en tu CSV)
    col_l = 'Goles_L' if 'Goles_L' in df.columns else 'Puntos_L'
    col_v = 'Goles_V' if 'Goles_V' in df.columns else 'Puntos_V'
    
    media_goles_local_liga = df[col_l].mean()
    media_goles_visita_liga = df[col_v].mean()
    
    # Prevención de división por cero en casos extremos
    media_goles_local_liga = max(media_goles_local_liga, 0.01)
    media_goles_visita_liga = max(media_goles_visita_liga, 0.01)

    # 2. Extraer historial de los equipos involucrados
    df_local_como_local = df[df['Local'] == equipo_local]
    df_local_como_visita = df[df['Visitante'] == equipo_local]
    
    df_visita_como_local = df[df['Local'] == equipo_visita]
    df_visita_como_visita = df[df['Visitante'] == equipo_visita]

    # Promedios del Equipo Local
    avg_goles_anotados_local = df_local_como_local[col_l].mean() if not df_local_como_local.empty else media_goles_local_liga
    avg_goles_recibidos_local = df_local_como_local[col_v].mean() if not df_local_como_local.empty else media_goles_visita_liga
    
    # Promedios del Equipo Visitante
    avg_goles_anotados_visita = df_visita_como_visita[col_v].mean() if not df_visita_como_visita.empty else media_goles_visita_liga
    avg_goles_recibidos_visita = df_visita_como_visita[col_l].mean() if not df_visita_como_visita.empty else media_goles_local_liga

    # 3. Calcular Fuerza de Ataque (Ratio vs Liga)
    fuerza_ataque_local = avg_goles_anotados_local / media_goles_local_liga
    fuerza_ataque_visita = avg_goles_anotados_visita / media_goles_visita_liga

    # 4. Calcular Fuerza de Defensa (Ratio vs Liga)
    # Una defensa fuerte tendrá un valor menor a 1.0 (permite menos goles que el promedio)
    fuerza_defensa_local = avg_goles_recibidos_local / media_goles_visita_liga
    fuerza_defensa_visita = avg_goles_recibidos_visita / media_goles_local_liga

    # 5. Cálculo Final de Lambdas Cruzadas
    lambda_local = fuerza_ataque_local * fuerza_defensa_visita * media_goles_local_liga
    lambda_visita = fuerza_ataque_visita * fuerza_defensa_local * media_goles_visita_liga

    # Si tienes variables de altitud o impacto arbitral, puedes multiplicarlas aquí.
    # Ejemplo: lambda_local *= factor_altitud
    
    return max(lambda_local, 0.1), max(lambda_visita, 0.1)

def simular_partido_montecarlo(equipo_local, equipo_visita):
    # Cargar el histórico directamente o pasarlo como parámetro
    try:
        df_historico = pd.read_csv('data/historico_liga_mx.csv')
    except FileNotFoundError:
        # Fallback en caso de que la ruta varíe (en local o en nube)
        df_historico = pd.read_csv('liga_mx_analytics/data/historico_liga_mx.csv')

    # 1. Obtener Lambdas Ajustadas por Fuerza del Oponente
    lambda_local, lambda_visita = calcular_lambdas_ajustadas(df_historico, equipo_local, equipo_visita)
    
    n_sims = 10000
    
    # 2. Generar simulación cruda de Poisson
    goles_l = np.random.poisson(lambda_local, n_sims)
    goles_v = np.random.poisson(lambda_visita, n_sims)

    # ... [El resto de tu código de Dixon-Coles continúa exactamente igual a partir de aquí] ...
    
    # 3. Calcular matriz probabilística inicial cruda
    resultados_exactos = {}
    for i in range(n_sims):
        marcador = f"{goles_l[i]}-{goles_v[i]}"
        resultados_exactos[marcador] = resultados_exactos.get(marcador, 0) + 1
        
    for k in resultados_exactos.keys():
        resultados_exactos[k] = resultados_exactos[k] / n_sims
        
    # 4. APLICAR CORRECCIÓN DIXON-COLES
    # El valor de rho=-0.15 es un estándar fuerte para el fútbol moderno de baja anotación
    matriz_corregida = aplicar_dixon_coles(lambda_local, lambda_visita, resultados_exactos, rho=-0.15)
    
    # 5. Reconstruir 1X2 basándose en la matriz corregida
    prob_local = 0.0
    prob_visita = 0.0
    prob_empate = 0.0
    prob_over = 0.0
    
    for marcador, prob in matriz_corregida.items():
        gl, gv = map(int, marcador.split('-'))
        if gl > gv:
            prob_local += prob
        elif gv > gl:
            prob_visita += prob
        else:
            prob_empate += prob
            
        if (gl + gv) > 2.5:
            prob_over += prob
            
    # Formatear la salida esperada por tu app.py y módulos existentes
    # (Mantén aquí el formato de retorno exacto que tu código ya utilizaba, 
    # añadiendo corners y tarjetas si los estabas simulando de forma similar).
    
    return {
        "Resultado_1X2": {
            "Gana Local": round(prob_local * 100, 1),
            "Empate": round(prob_empate * 100, 1),
            "Gana Visita": round(prob_visita * 100, 1)
        },
        "Goles_Over_Under": {
            "Over 2.5": round(prob_over * 100, 1)
        },
        # Asegúrate de mantener la estructura de Goles, Corners y Tarjetas Individuales
        # que tu app.py lee para que la interfaz no marque errores.
    }
            
            "Under 4.5 Tarjetas": round(np.sum(total_tarjetas <= 4.5) / n_simulaciones * 100, 2) # NUEVO
        }
    }
    
    return resultados
