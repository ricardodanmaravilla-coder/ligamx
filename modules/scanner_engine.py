import os
import requests
import pandas as pd
from modules.stats_engine import calcular_expectativa_partido
from modules.montecarlo_sim import simular_partido_montecarlo
from modules.odds_engine import obtener_cuotas_partido, evaluar_mercado_avanzado
from modules.ml_engine import PredictorML
from datetime import datetime
from github import Github
import streamlit as st
import io

# Configuración API-Sports
API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}
LIGA_MX_ID = 262

def registrar_apuesta_github(partido, mercado, prob_modelo, cuota, kelly_pct, bankroll_inicial=5000):
    """
    Calcula la inversión, crea el registro y hace un commit automático al repositorio de GitHub.
    """
    if kelly_pct <= 0:
        return

    token = st.secrets["MI_GITHUB_TOKEN"]
    g = Github(token)
    nombre_repo = "ricardodanmaravilla-coder/ligamx" 
    repo = g.get_repo(nombre_repo)
    ruta_archivo_github = 'data/registro_apuestas.csv'
    
    porcentaje_seguro = min(kelly_pct, 10.0) 
    inversion_mxn = round(bankroll_inicial * (porcentaje_seguro / 100), 2)
    ganancia_potencial = round((inversion_mxn * cuota) - inversion_mxn, 2)
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    nueva_fila_df = pd.DataFrame([{
        "Fecha_Registro": fecha_actual,
        "Partido": partido,
        "Mercado": mercado,
        "Prob_Modelo_%": prob_modelo,
        "Cuota_Apostada": cuota,
        "Kelly_%": round(porcentaje_seguro, 2),
        "Inversion_MXN": inversion_mxn,
        "Ganancia_Potencial_MXN": ganancia_potencial,
        "Estado": "Pendiente",
        "Retorno_Real_MXN": 0.0
    }])

    try:
        archivo_en_repo = repo.get_contents(ruta_archivo_github)
        contenido_actual = archivo_en_repo.decoded_content.decode('utf-8')
        df_existente = pd.read_csv(io.StringIO(contenido_actual))
        
        duplicado = df_existente[(df_existente['Partido'] == partido) & (df_existente['Mercado'] == mercado) & (df_existente['Estado'] == 'Pendiente')]
        if not duplicado.empty:
            return

        df_final = pd.concat([df_existente, nueva_fila_df], ignore_index=True)
        nuevo_contenido_csv = df_final.to_csv(index=False)
        
        repo.update_file(
            path=archivo_en_repo.path,
            message=f"🤖 Registro automático de apuesta: {mercado}",
            content=nuevo_contenido_csv,
            sha=archivo_en_repo.sha
        )
    except Exception as e:
        nuevo_contenido_csv = nueva_fila_df.to_csv(index=False)
        repo.create_file(
            path=ruta_archivo_github,
            message="🤖 Creación de bitácora inicial",
            content=nuevo_contenido_csv
        )
        
def escanear_jornada_actual(temporada_actual=2026):
    """
    Escanea la jornada cruzando Montecarlo y Machine Learning. 
    Filtra y recomienda únicamente los mercados donde ambos modelos coinciden en alta probabilidad.
    """
    
    # 1. Entrenamos el motor de Machine Learning multimodelo con el histórico real
    ml_escanner = PredictorML()
    df_historico_ml = None
    try:
        df_historico_ml = pd.read_csv("data/historico_ligamx_completo.csv")
        df_historico_ml['Local'] = df_historico_ml['Local'].str.strip()
        df_historico_ml['Visitante'] = df_historico_ml['Visitante'].str.strip()
        ml_escanner.entrenar(df_historico_ml)
    except Exception:
        pass

    # 2. Obtenemos la ronda activa actual de la Liga MX
    url_rounds = f"{BASE_URL}/fixtures/rounds"
    res_rounds = requests.get(url_rounds, headers=HEADERS, params={"league": LIGA_MX_ID, "season": temporada_actual, "current": "true"})
    
    jornada_actual = None
    if res_rounds.status_code == 200:
        rounds_data = res_rounds.json().get("response", [])
        if rounds_data:
            jornada_actual = rounds_data[0]
            
    url = f"{BASE_URL}/fixtures"
    params = {"league": LIGA_MX_ID, "season": temporada_actual, "status": "NS"}
    if jornada_actual:
        params["round"] = jornada_actual
    
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code != 200:
        return []
        
    fixtures = res.json().get("response", [])
    if not fixtures:
        params = {"league": LIGA_MX_ID, "season": temporada_actual, "status": "NS"}
        res = requests.get(url, headers=HEADERS, params=params)
        if res.status_code == 200:
            fixtures = res.json().get("response", [])

    oportunidades_oro = []
    
    mercados_a_mapear = [
        ("Gana Local", "1"),
        ("Gana Visita", "2"),
        ("Over 2.5 Goles", "Over 2.5"),
        ("Over 9.5 Corners", "Over 9.5 Corners")
    ]

    for p in fixtures:
        fix_id = p["fixture"]["id"]
        local = p["teams"]["home"]["name"]
        visita = p["teams"]["away"]["name"]
        fecha = p["fixture"]["date"][:16].replace("T", " ")
        
        try:
            # 3. Corremos Montecarlo
            resultados = simular_partido_montecarlo(local, visita)
            if isinstance(resultados, str): continue
            
            # 4. Obtenemos cuotas automáticas
            cuotas = obtener_cuotas_partido(fix_id)
            if not cuotas: continue
            
            # Probabilidades de Montecarlo
            prob_mc_local = resultados["Resultado_1X2"]["Gana Local"]
            prob_mc_visita = resultados["Resultado_1X2"]["Gana Visita"]
            prob_mc_goles = resultados["Goles_Over_Under"]["Over 2.5"]
            prob_mc_corners = resultados["Corners_Totales"]["Over 9.5 Corners"]

            # Probabilidades de Machine Learning (si está entrenado)
            prob_ml_local = 0.0
            prob_ml_visita = 0.0
            prob_ml_goles = 0.0
            prob_ml_corners = 0.0

            if ml_escanner.is_trained and df_historico_ml is not None:
                g_l_sim = resultados.get('Goles_Individuales', {}).get(local, {}).get('goles', 1.2)
                g_v_sim = resultados.get('Goles_Individuales', {}).get(visita, {}).get('goles', 1.0)
                
                preds_ml = ml_escanner.predecir_mercados_completos(df_historico_ml, local, visita, g_l_sim, g_v_sim)
                prob_ml_local = preds_ml['1X2']['Gana Local']
                prob_ml_visita = preds_ml['1X2']['Gana Visita']
                prob_ml_goles = preds_ml['Over_2.5_Goles']
                prob_ml_corners = preds_ml['Over_9.5_Corners']

            # Diccionarios para evaluar en bucle de consenso
            prob_mc_dict = {
                "Gana Local": prob_mc_local,
                "Gana Visita": prob_mc_visita,
                "Over 2.5 Goles": prob_mc_goles,
                "Over 9.5 Corners": prob_mc_corners
            }

            prob_ml_dict = {
                "Gana Local": prob_ml_local,
                "Gana Visita": prob_ml_visita,
                "Over 2.5 Goles": prob_ml_goles,
                "Over 9.5 Corners": prob_ml_corners
            }

            llaves_mercado = {
                "Gana Local": "1",
                "Gana Visita": "2",
                "Over 2.5 Goles": "Over 2.5",
                "Over 9.5 Corners": "Over 9.5 Corners"
            }

           # 5. FILTRO DE CONSENSO MAESTRO (MONTECARLO + MACHINE LEARNING) - EL PUNTO DULCE
            for nombre_m, llave in mercados_a_mapear:
                p_mc = prob_mc_dict[nombre_m]
                p_ml = prob_ml_dict[nombre_m]
                cuota = cuotas.get(llave)
                
                # UMBRAL BALANCEADO: Ambos modelos deben marcar al menos 58% (Consenso sólido)
                # Y la cuota no debe ser basura (mayor a 1.40)
                if cuota and cuota > 1.40 and p_mc >= 58.0 and p_ml >= 58.0:
                    
                    prob_combinada = round((p_mc + p_ml) / 2, 1)
                    _, ev, veredicto, stake, riesgo = evaluar_mercado_avanzado(prob_combinada, cuota)
                    
                    # FILTRO DE VALOR REALISTA: Solo EV mayor a 2% (Ventaja estadística limpia)
                    if ev >= 2.0:
                        oportunidades_oro.append({
                            "Fecha": fecha,
                            "Partido": f"{local} vs {visita}",
                            "Mercado": nombre_m,
                            "P. Montecarlo": f"{p_mc}%",
                            "P. ML": f"{p_ml}%",
                            "Cuota": f"{cuota:.2f}",
                            "EV (Valor)": f"+{ev:.1f}%",
                            "Riesgo": riesgo,
                            "Stake Rec.": f"{stake:.1f}%",
                            "Veredicto": f"✅ CONSENSO DE VALOR ({veredicto})",
                            "Fixture_ID": fix_id
                        })
        except Exception as e:
            continue
            
    return oportunidades_oro
