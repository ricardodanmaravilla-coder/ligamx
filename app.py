import os
import datetime
import pandas as pd
import requests
import streamlit as st

from modules.feature_engineering import clean_history, normalize_team
from modules.elo_engine import SistemaEloLigaMX
from modules.ml_engine import PredictorML
from modules.montecarlo_sim import simular_partido_montecarlo
from modules.odds_engine import obtener_cuotas_partido
from modules.scanner_engine import evaluar_fixture, escanear_jornada_actual

st.set_page_config(page_title="Liga MX Analytics V3", layout="wide")
st.title("Liga MX Analytics — Model V3")
st.caption("API-Football · ELO prepartido · rolling features · ML · Monte Carlo · value betting 1X2 + O/U")

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LIGA_MX_ID = 262
LIGA_MX_SEASON = 2026

@st.cache_data(ttl=3600)
def cargar_historico():
    for path in ["data/historico_ligamx_completo.csv", "historico_ligamx_completo.csv"]:
        try:
            return clean_history(pd.read_csv(path))
        except Exception:
            pass
    raise RuntimeError("No se pudo cargar el historico")

@st.cache_data(ttl=300)
def proximos_partidos():
    if not API_KEY:
        return [], "Falta la variable API_SPORTS_KEY"
    hoy = datetime.date.today().strftime("%Y-%m-%d")
    futuro = (datetime.date.today() + datetime.timedelta(days=45)).strftime("%Y-%m-%d")
    params = {"league": LIGA_MX_ID, "season": LIGA_MX_SEASON, "from": hoy, "to": futuro, "timezone": "America/Mexico_City"}
    try:
        r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params, timeout=12)
    except Exception as exc:
        return [], f"Error de conexión con API-Football: {exc}"
    if r.status_code != 200:
        return [], f"API-Football HTTP {r.status_code}: {r.text[:250]}"
    try:
        payload = r.json()
    except Exception:
        return [], "API-Football devolvió una respuesta que no es JSON"
    if payload.get("errors"):
        return [], f"API-Football reportó error: {payload.get('errors')}"
    out=[]
    for f in payload.get("response", []):
        status=f.get("fixture",{}).get("status",{}).get("short","")
        if status in {"FT","AET","PEN","CANC","ABD","PST"}:
            continue
        out.append({
            "fixture_id":f["fixture"]["id"],
            "fecha":f["fixture"]["date"][:16].replace("T"," "),
            "local":normalize_team(f["teams"]["home"]["name"]),
            "visita":normalize_team(f["teams"]["away"]["name"]),
        })
    if not out:
        return [], f"API-Football respondió correctamente pero no devolvió próximos partidos. results={payload.get('results',0)}"
    return out, None


def line_or_none(cuotas, kind):
    value=cuotas.get(f"linea_{kind}_detectada") if cuotas else None
    return float(value) if value is not None else None


def metric_market(container,title,mc_section,ml_section,over_key,under_key,push_key=None):
    mo,mu=mc_section.get(over_key),mc_section.get(under_key)
    lo,lu=ml_section.get(over_key),ml_section.get(under_key)
    if mo is None or mu is None or lo is None or lu is None:
        container.info(f"{title}: sin predicción completa")
        return
    mp=mc_section.get(push_key,0.0) if push_key else 0.0
    lp=ml_section.get(push_key,0.0) if push_key else 0.0
    container.markdown(f"**{title}**")
    container.write(f"Monte Carlo: Over {mo:.1f}% · Under {mu:.1f}%" + (f" · Push {mp:.1f}%" if mp else ""))
    container.write(f"ML: Over {lo:.1f}% · Under {lu:.1f}%" + (f" · Push {lp:.1f}%" if lp else ""))

try:
    df=cargar_historico()
except Exception as exc:
    st.error(str(exc)); st.stop()

st.sidebar.header("Estado del modelo")
st.sidebar.metric("Partidos historicos",f"{len(df):,}")
st.sidebar.write(f"Desde **{df.Fecha.min().date()}** hasta **{df.Fecha.max().date()}**")
st.sidebar.write(f"Liga MX API: **league {LIGA_MX_ID} · season {LIGA_MX_SEASON}**")
st.sidebar.write("1X2 = validado OOS. O/U V3 = integrado con filtros más estrictos y diagnóstico separado.")
st.sidebar.warning("O/U V3 todavía no debe interpretarse con la misma confianza histórica que 1X2.")

fixtures,fixtures_error=proximos_partidos()
if not API_KEY:
    st.warning("No se encontró API_SPORTS_KEY.")
elif fixtures_error:
    st.error(fixtures_error)
elif not fixtures:
    st.warning("No se encontraron próximos partidos con API-Football.")
else:
    labels=[f"{p['fecha']} | {p['local']} vs {p['visita']}" for p in fixtures]
    choice=st.selectbox("Selecciona un partido",range(len(fixtures)),format_func=lambda i:labels[i])
    partido=fixtures[choice]

    if st.button("Analizar partido V3",type="primary"):
        local,visita,fixture_id=partido["local"],partido["visita"],partido["fixture_id"]
        with st.spinner("Calculando ELO, Monte Carlo, ML, líneas y cuotas reales..."):
            try:
                tabla_elo=SistemaEloLigaMX().calcular_historico(df)
                elo_map=dict(zip(tabla_elo.Equipo,tabla_elo.ELO_Rating))
                if local not in elo_map or visita not in elo_map:
                    raise ValueError("Equipo sin ELO historico suficiente: NO BET")

                cuotas=obtener_cuotas_partido(fixture_id)
                lg,lc,lt=line_or_none(cuotas,"goles"),line_or_none(cuotas,"corners"),line_or_none(cuotas,"tarjetas")
                tech_g,tech_c,tech_t=lg if lg is not None else 2.5,lc if lc is not None else 9.5,lt if lt is not None else 4.5

                mc=simular_partido_montecarlo(local,visita,df_historico=df,elo_local=elo_map[local],elo_visita=elo_map[visita],linea_goles=tech_g,linea_corners=tech_c,linea_tarjetas=tech_t)
                ml=PredictorML()
                if not ml.entrenar(df):
                    raise ValueError("El ML no pudo entrenar: NO BET")
                mlp=ml.predecir_mercados_completos(df,local,visita,elo_local=elo_map[local],elo_visita=elo_map[visita],linea_goles=tech_g,linea_corners=tech_c,linea_tarjetas=tech_t)

                st.subheader(f"{local} vs {visita}")
                c1,c2,c3=st.columns(3)
                c1.metric("ELO local",f"{elo_map[local]:.1f}")
                c2.metric("ELO visitante",f"{elo_map[visita]:.1f}")
                c3.metric("Diferencia ELO",f"{elo_map[local]-elo_map[visita]:+.1f}")

                st.markdown("### 1X2")
                rows=[]
                for market in ["Gana Local","Empate","Gana Visita"]:
                    rows.append({"Mercado":market,"Monte Carlo %":mc["Resultado_1X2"][market],"ML %":mlp["Resultado_1X2"][market],"Diferencia pp":round(abs(mc["Resultado_1X2"][market]-mlp["Resultado_1X2"][market]),1)})
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

                st.markdown("### O/U — goles, córners y tarjetas")
                gcol,ccol,tcol=st.columns(3)
                if lg is None:
                    gcol.info("Goles: sin línea real")
                else:
                    metric_market(gcol,f"Goles {lg}",mc["Goles_Over_Under"],mlp["Goles_Over_Under"],f"Over {lg}",f"Under {lg}",f"Push {lg}")
                if lc is None:
                    ccol.info("Corners: sin línea real")
                else:
                    metric_market(ccol,f"Corners {lc}",mc["Corners_Totales"],mlp["Corners_Totales"],f"Over {lc} Corners",f"Under {lc} Corners",f"Push {lc} Corners")
                if lt is None:
                    tcol.info("Tarjetas: sin línea real")
                else:
                    metric_market(tcol,f"Tarjetas {lt}",mc["Tarjetas_Totales"],mlp["Tarjetas_Totales"],f"Over {lt} Tarjetas",f"Under {lt} Tarjetas",f"Push {lt} Tarjetas")

                st.markdown("### Picks del modelo")
                picks,diagnostics=evaluar_fixture(local,visita,fixture_id,df,cuotas=cuotas,ml=ml,elo_map=elo_map,return_diagnostics=True)
                if picks:
                    st.success(f"{len(picks)} oportunidad(es) superaron todos los filtros")
                    st.dataframe(pd.DataFrame(picks),use_container_width=True,hide_index=True)
                else:
                    st.info("NO BET — ninguna opción superó todos los filtros.")

                st.markdown("#### Diagnóstico completo 1X2 + O/U")
                if diagnostics:
                    st.dataframe(pd.DataFrame(diagnostics),use_container_width=True,hide_index=True)

                with st.expander("Detalle técnico"):
                    st.write("Cuotas observadas",cuotas)
                    st.write("Totales previstos por ML",mlp.get("Prediccion_Totales",{}))
                    st.write("Expectativas Monte Carlo",{
                        "goles_local":mc["Goles_Individuales"][local]["goles"],
                        "goles_visita":mc["Goles_Individuales"][visita]["goles"],
                        "corners_local":mc["Corners_Individuales"][local]["corners"],
                        "corners_visita":mc["Corners_Individuales"][visita]["corners"],
                    })
            except Exception as exc:
                st.error(f"NO BET / error de validación: {exc}")

st.markdown("---")
st.subheader("Scanner de jornada V3 — 1X2 + O/U")
st.caption("1X2 usa filtros V2 validados. O/U V3 exige línea y ambas cuotas reales, no-vig, probabilidad, edge, EV y acuerdo entre modelos.")
if st.button("Escanear jornada completa V3"):
    with st.spinner("Escaneando próximos partidos..."):
        picks=escanear_jornada_actual(df)
    if picks:
        st.dataframe(pd.DataFrame(picks),use_container_width=True,hide_index=True)
    else:
        st.info("NO BET — no hay oportunidades que cumplan todos los filtros.")

st.markdown("---")
st.subheader("Ranking ELO actual")
st.dataframe(SistemaEloLigaMX().calcular_historico(df),use_container_width=True,hide_index=True)
