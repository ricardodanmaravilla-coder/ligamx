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

def simular_partido_montecarlo(equipo_local, equipo_visita, df_historico=None):
    """
    Ejecuta 10,000 iteraciones de Montecarlo.
    Asegúrate de pasar tu df_historico u obtenerlo dentro de la función.
    """
    # 1. Obtener Lambdas (Asegúrate de usar tus funciones de promedio, altitud, etc. aquí)
    # Por ejemplo, si tienes tu función que ya calcula lambda_local y lambda_visita:
    # lambda_local = calcular_lambda_local(...)
    # lambda_visita = calcular_lambda_visita(...)
    
    # Valores de ejemplo de respaldo (reemplaza con tus variables de altitud/histórico)
    lambda_local = 1.45 
    lambda_visita = 1.15
    n_sims = 10000
    
    # 2. Generar simulación cruda de Poisson
    goles_l = np.random.poisson(lambda_local, n_sims)
    goles_v = np.random.poisson(lambda_visita, n_sims)
    
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
