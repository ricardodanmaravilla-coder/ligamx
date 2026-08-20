import os
import requests
import pandas as pd
from datetime import datetime
from github import Github
import streamlit as st
import io
import time

from modules.stats_engine import calcular_expectativa_partido
from modules.montecarlo_sim import simular_partido_montecarlo
from modules.odds_engine import obtener_cuotas_partido, evaluar_mercado_avanzado
from modules.ml_engine import PredictorML

API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}
LIGA_MX_ID = 262

def registrar_apuesta_github(partido, mercado, prob_modelo, cuota, kelly_pct, bankroll_inicial=5000):
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

def obtener_ultimo_elo(df, equipo):
    try:
        if df is not None and not df.empty:
            df_eq = df[(df['Local'] == equipo) | (df['Visitante'] == equipo)]
            if not df_eq.empty:
                ultima_fila = df_eq.iloc[-1]
                if ultima_fila['Local'] == equipo:
                    for col in ['ELO_Local', 'ELO_L', 'Elo_Local']:
                        if col in df.columns and not pd.isna(ultima_fila[col]):
                            return float(ultima_fila[col])
                else:
                    for col in ['ELO_Visita', 'ELO_V', 'Elo_Visita', 'ELO_Visitante']:
                        if col in df.columns and not pd.isna(ultima_fila[col]):
                            return float(ultima_fila[col])
    except:
        pass
    return 1500.0

def _obtener_partidos_escanner(league_id, temporada_actual):
    """Busca los partidos de la jornada con rango de fechas y respaldo de ESPN."""
    url = f"{BASE_URL}/fixtures"
    import datetime
    
    hoy = datetime.date.today().strftime("%Y-%m-%d")
    futuro = (datetime.date.today() + datetime.timedelta(days=45)).strftime("%Y-%m-%d")
    
    querystring = {"league": str(league_id), "from": hoy, "to": futuro}
    response = requests.get(url, headers=HEADERS, params=querystring)
    datos = []
    
    if response.status_code == 200:
        datos = response.json().get("response", [])
        
    if not datos:
        querystring_fallback = {"league": str(league_id), "next": "15"}
        response = requests.get(url, headers=HEADERS, params=querystring_fallback)
        if response.status_code == 200:
            datos = response.json().get("response", [])

    partidos_list = []
    for p in datos:
        estado = p.get("fixture", {}).get("status", {}).get("short", "")
        if estado in ["FT", "AET", "PEN", "CANC", "ABD"]: 
            continue
        partidos_list.append({
            "local": p["teams"]["home"]["name"],
            "visita": p["teams"]["away"]["name"],
            "fixture_id": p["fixture"]["id"],
            "fecha": p["fixture"]["date"][:16].replace("T", " ")
        })
        
    # --- ACTIVACIÓN DEL RESPALDO DE ESPN PARA EL ESCÁNER ---
    if not partidos_list:
        url_espn = "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard"
        try:
            res = requests.get(url_espn, timeout=5)
            if res.status_code == 200:
                for event in res.json().get("events", []):
                    estado = event.get("status", {}).get("type", {}).get("name", "")
                    if estado in ["STATUS_FINAL", "STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_FULL_TIME"]: 
                        continue
                        
                    comp = event.get("competitions", [])[0]
                    local, visita = "Local", "Visita"
                    for team in comp.get("competitors", []):
                        if team.get("homeAway") == "home": 
                            local = team.get("team", {}).get("name", "Local")
                        else: 
                            visita = team.get("team", {}).get("name", "Visita")
                    
                    partidos_list.append({
                        "local": local,
                        "visita": visita,
                        "fixture_id": event.get("id"),
                        "fecha": event.get("date", "")[:16].replace("T", " ")
                    })
        except Exception as e:
            print(f"Fallo en respaldo ESPN Escáner: {e}")
            
    return partidos_list

def escanear_jornada_actual(temporada_actual=2026):
    """Mantiene compatibilidad exacta por defecto con Liga MX"""
    return escanear_jornada_personalizada(league_id=LIGA_MX_ID, ruta_csv='data/historico_ligamx_completo.csv', temporada_actual=temporada_actual)

def escanear_jornada_personalizada(league_id, ruta_csv, temporada_actual=2026):
    """Escáner flexible adaptado para cualquier liga (Liga MX o Leagues Cup)"""
    ml_escanner = PredictorML()
    df_historico_ml = None
    
    url_github_raw = f'https://raw.githubusercontent.com/ricardodanmaravilla-coder/ligamx/main/{ruta_csv}'
    rutas_locales = [ruta_csv, os.path.basename(ruta_csv)]
    
    try:
        for r in rutas_locales:
            if os.path.exists(r):
                df_historico_ml = pd.read_csv(r)
                break
        if df_historico_ml is None:
            df_historico_ml = pd.read_csv(url_github_raw)
            
        df_historico_ml['Local'] = df_historico_ml['Local'].str.strip()
        df_historico_ml['Visitante'] = df_historico_ml['Visitante'].str.strip()
        ml_escanner.entrenar(df_historico_ml)
    except Exception:
        pass

    fixtures = _obtener_partidos_escanner(league_id, temporada_actual)
    if not fixtures:
        return []

    oportunidades_oro = []

    st.write("---")
    st.write(f"🔎 **Escaneando partidos oficiales (Liga ID: {league_id}) y validando modelos...**")

    for p in fixtures:
        try:
            fix_id = p["fixture_id"]
            local = p["local"]
            visita = p["visita"]
            fecha = p["fecha"]
            
            st.write(f"⚙️ Analizando: **{local} vs {visita}**")
            
            cuotas = obtener_cuotas_partido(fix_id)
            if not cuotas: 
                st.write(f"⚠️ *Sin cuotas disponibles en este momento.*")
                continue
                
            # Extracción dinámica de las líneas reales detectadas del casino
            l_goles = cuotas.get("linea_goles_detectada", "2.5")
            l_corners = cuotas.get("linea_corners_detectada", "9.5")
            l_tarjetas = cuotas.get("linea_tarjetas_detectada", "4.5")
                
            elo_loc = obtener_ultimo_elo(df_historico_ml, local)
            elo_vis = obtener_ultimo_elo(df_historico_ml, visita)
            
            resultados = simular_partido_montecarlo(
                local, visita, 
                df_historico=df_historico_ml, 
                elo_local=elo_loc, 
                elo_visita=elo_vis,
                linea_goles=float(l_goles),
                linea_corners=float(l_corners),
                linea_tarjetas=float(l_tarjetas)
            )
            if isinstance(resultados, str): 
                continue

            prob_ml_local = prob_ml_empate = prob_ml_visita = 0.0
            prob_ml_over_g = prob_ml_under_g = 0.0
            prob_ml_over_c = prob_ml_under_c = 0.0
            prob_ml_over_t = prob_ml_under_t = 0.0

            if ml_escanner.is_trained and df_historico_ml is not None:
                g_l_sim = resultados.get('Goles_Individuales', {}).get(local, {}).get('goles', 1.2)
                g_v_sim = resultados.get('Goles_Individuales', {}).get(visita, {}).get('goles', 1.0)
                
                preds_ml = ml_escanner.predecir_mercados_completos(
                    df_historico_ml, local, visita, g_l_sim, g_v_sim, elo_loc, elo_vis,
                    linea_goles=float(l_goles), linea_corners=float(l_corners), linea_tarjetas=float(l_tarjetas)
                )
                
                if "Resultado_1X2" in preds_ml:
                    prob_ml_local = preds_ml['Resultado_1X2']['Gana Local']
                    prob_ml_empate = preds_ml['Resultado_1X2']['Empate']
                    prob_ml_visita = preds_ml['Resultado_1X2']['Gana Visita']
                    
                    prob_ml_over_g = preds_ml['Goles_Over_Under'].get(f'Over {l_goles}', 0.0)
                    prob_ml_under_g = preds_ml['Goles_Over_Under'].get(f'Under {l_goles}', 0.0)
                    
                    prob_ml_over_c = preds_ml['Corners_Totales'].get(f'Over {l_corners} Corners', 0.0)
                    prob_ml_under_c = preds_ml['Corners_Totales'].get(f'Under {l_corners} Corners', 0.0)
                    
                    prob_ml_over_t = preds_ml['Tarjetas_Totales'].get(f'Over {l_tarjetas} Tarjetas', 0.0)
                    prob_ml_under_t = preds_ml['Tarjetas_Totales'].get(f'Under {l_tarjetas} Tarjetas', 0.0)

            prob_mc_dict = {
                "Gana Local": resultados.get('Resultado_1X2', {}).get('Gana Local', 0.0),
                "Empate": resultados.get('Resultado_1X2', {}).get('Empate', 0.0),
                "Gana Visita": resultados.get('Resultado_1X2', {}).get('Gana Visita', 0.0),
                f"Over {l_goles} Goles": resultados.get('Goles_Over_Under', {}).get(f'Over {l_goles}', 0.0),
                f"Under {l_goles} Goles": resultados.get('Goles_Over_Under', {}).get(f'Under {l_goles}', 0.0),
                f"Over {l_corners} Corners": resultados.get('Corners_Totales', {}).get(f'Over {l_corners} Corners', 0.0),
                f"Under {l_corners} Corners": resultados.get('Corners_Totales', {}).get(f'Under {l_corners} Corners', 0.0),
                f"Over {l_tarjetas} Tarjetas": resultados.get('Tarjetas_Totales', {}).get(f'Over {l_tarjetas} Tarjetas', 0.0),
                f"Under {l_tarjetas} Tarjetas": resultados.get('Tarjetas_Totales', {}).get(f'Under {l_tarjetas} Tarjetas', 0.0)
            }

            prob_ml_dict = {
                "Gana Local": prob_ml_local,
                "Empate": prob_ml_empate,
                "Gana Visita": prob_ml_visita,
                f"Over {l_goles} Goles": prob_ml_over_g,
                f"Under {l_goles} Goles": prob_ml_under_g,
                f"Over {l_corners} Corners": prob_ml_over_c,
                f"Under {l_corners} Corners": prob_ml_under_c,
                f"Over {l_tarjetas} Tarjetas": prob_ml_over_t,
                f"Under {l_tarjetas} Tarjetas": prob_ml_under_t
            }

            mercados_a_mapear = [
                ("Gana Local", "1"),
                ("Empate", "X"),
                ("Gana Visita", "2"),
                (f"Over {l_goles} Goles", f"Over {l_goles}"),
                (f"Under {l_goles} Goles", f"Under {l_goles}"),
                (f"Over {l_corners} Corners", f"Over {l_corners} Corners"),
                (f"Under {l_corners} Corners", f"Under {l_corners} Corners"),
                (f"Over {l_tarjetas} Tarjetas", f"Over {l_tarjetas} Tarjetas"),
                (f"Under {l_tarjetas} Tarjetas", f"Under {l_tarjetas} Tarjetas")
            ]

            for nombre_m, llave_api in mercados_a_mapear:
                try:
                    p_mc = float(prob_mc_dict.get(nombre_m, 0.0))
                except:
                    p_mc = 0.0
                    
                try:
                    p_ml = float(prob_ml_dict.get(nombre_m, 0.0))
                except:
                    p_ml = 0.0
                
                raw_cuota = cuotas.get(llave_api)
                try:
                    cuota = float(raw_cuota) if raw_cuota is not None else 0.0
                except:
                    cuota = 0.0

                if cuota > 1.40 and p_mc >= 58.0 and p_ml >= 58.0:
                    prob_combinada = round((p_mc + p_ml) / 2.0, 1)
                    
                    prob_implicita_casa = (1.0 / cuota) * 100.0
                    diferencia_anomala = prob_combinada - prob_implicita_casa
                    
                    if diferencia_anomala > 40.0:
                        continue 
                    
                    ev_real = ((prob_combinada / 100.0) * cuota) - 1.0
                    ev_porcentaje = ev_real * 100.0
                    
                    if ev_porcentaje >= 2.0:
                        stake_recomendado = (ev_real / (cuota - 1.0)) * 10.0 
                        stake_recomendado = max(0.5, min(stake_recomendado, 3.0)) 
                        
                        oportunidades_oro.append({
                            "Fecha": fecha,
                            "Partido": f"{local} vs {visita}",
                            "Mercado": nombre_m,
                            "P. Montecarlo": f"{p_mc}%",
                            "P. ML": f"{p_ml}%",
                            "Cuota": f"{cuota:.2f}",
                            "EV (Valor)": f"+{ev_porcentaje:.1f}%",
                            "Stake Rec.": f"{stake_recomendado:.1f}%",
                            "Veredicto": "✅ CONSENSO BLINDADO",
                            "Fixture_ID": fix_id
                        })

        except Exception as e:
            continue
            
    st.write("✅ **Escaneo completado con éxito.**")
    return oportunidades_oro
