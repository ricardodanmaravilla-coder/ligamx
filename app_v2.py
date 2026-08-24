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


st.set_page_config(page_title="Liga MX Analytics V2", layout="wide")
st.title("Liga MX Analytics — Model V2")
st.caption("Datos API-Football · ELO prepartido · rolling features · ML calibrado · Monte Carlo · value betting")

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LIGA_MX_ID = 262


@st.cache_data(ttl=3600)
def cargar_historico():
    paths = ["data/historico_ligamx_completo.csv", "historico_ligamx_completo.csv"]
    for path in paths:
        try:
            return clean_history(pd.read_csv(path))
        except Exception:
            pass
    raise RuntimeError("No se pudo cargar el historico")


@st.cache_data(ttl=900)
def proximos_partidos():
    if not API_KEY:
        return []
    hoy = datetime.date.today().strftime("%Y-%m-%d")
    futuro = (datetime.date.today() + datetime.timedelta(days=45)).strftime("%Y-%m-%d")
    try:
        r = requests.get(
            f"{BASE_URL}/fixtures",
            headers=HEADERS,
            params={"league": LIGA_MX_ID, "from": hoy, "to": futuro},
            timeout=12,
        )
        if r.status_code != 200:
            return []
        data = r.json().get("response", [])
    except Exception:
        return []

    out = []
    for f in data:
        status = f.get("fixture", {}).get("status", {}).get("short", "")
        if status in {"FT", "AET", "PEN", "CANC", "ABD", "PST"}:
            continue
        out.append({
            "fixture_id": f["fixture"]["id"],
            "fecha": f["fixture"]["date"][:16].replace("T", " "),
            "local": normalize_team(f["teams"]["home"]["name"]),
            "visita": normalize_team(f["teams"]["away"]["name"]),
        })
    return out


def line_or_none(cuotas, kind):
    value = cuotas.get(f"linea_{kind}_detectada") if cuotas else None
    return float(value) if value is not None else None


def metric_market(container, title, section, over_key, under_key, push_key=None):
    over = section.get(over_key)
    under = section.get(under_key)
    push = section.get(push_key, 0.0) if push_key else 0.0
    if over is None or under is None:
        container.info(f"{title}: sin linea real disponible")
        return
    delta = f"Under {under:.1f}%"
    if push:
        delta += f" · Push {push:.1f}%"
    container.metric(title, f"Over {over:.1f}%", delta)


try:
    df = cargar_historico()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.header("Estado del modelo")
st.sidebar.metric("Partidos historicos", f"{len(df):,}")
st.sidebar.write(f"Desde **{df.Fecha.min().date()}** hasta **{df.Fecha.max().date()}**")
st.sidebar.write("Regla de seguridad: si falta cuota, equipo o muestra suficiente → **NO BET**")

fixtures = proximos_partidos()

if not API_KEY:
    st.warning("No se encontro API_SPORTS_KEY. La V2 no inventa fixtures ni cuotas.")
elif not fixtures:
    st.warning("No se encontraron proximos partidos con API-Football.")
else:
    labels = [f"{p['fecha']} | {p['local']} vs {p['visita']}" for p in fixtures]
    choice = st.selectbox("Selecciona un partido", range(len(fixtures)), format_func=lambda i: labels[i])
    partido = fixtures[choice]

    if st.button("Analizar partido V2", type="primary"):
        local = partido["local"]
        visita = partido["visita"]
        fixture_id = partido["fixture_id"]

        with st.spinner("Calculando ELO, Monte Carlo, ML y cuotas reales..."):
            try:
                tabla_elo = SistemaEloLigaMX().calcular_historico(df)
                elo_map = dict(zip(tabla_elo.Equipo, tabla_elo.ELO_Rating))
                if local not in elo_map or visita not in elo_map:
                    raise ValueError("Equipo sin ELO historico suficiente: NO BET")

                cuotas = obtener_cuotas_partido(fixture_id)
                lg = line_or_none(cuotas, "goles")
                lc = line_or_none(cuotas, "corners")
                lt = line_or_none(cuotas, "tarjetas")

                # Las lineas tecnicas solo permiten construir las estructuras internas.
                # Nunca se presentan como cuotas/mercados si la API no las entrego.
                tech_g = lg if lg is not None else 2.5
                tech_c = lc if lc is not None else 9.5
                tech_t = lt if lt is not None else 4.5

                mc = simular_partido_montecarlo(
                    local,
                    visita,
                    df_historico=df,
                    elo_local=elo_map[local],
                    elo_visita=elo_map[visita],
                    linea_goles=tech_g,
                    linea_corners=tech_c,
                    linea_tarjetas=tech_t,
                )

                ml = PredictorML()
                if not ml.entrenar(df):
                    raise ValueError("El ML no pudo entrenar: NO BET")
                mlp = ml.predecir_mercados_completos(
                    df,
                    local,
                    visita,
                    elo_local=elo_map[local],
                    elo_visita=elo_map[visita],
                    linea_goles=tech_g,
                    linea_corners=tech_c,
                    linea_tarjetas=tech_t,
                )

                st.subheader(f"{local} vs {visita}")
                c1, c2, c3 = st.columns(3)
                c1.metric("ELO local", f"{elo_map[local]:.1f}")
                c2.metric("ELO visitante", f"{elo_map[visita]:.1f}")
                c3.metric("Diferencia ELO", f"{elo_map[local]-elo_map[visita]:+.1f}")

                st.markdown("### 1X2 — comparación de modelos")
                rows = []
                for market in ["Gana Local", "Empate", "Gana Visita"]:
                    rows.append({
                        "Mercado": market,
                        "Monte Carlo %": mc["Resultado_1X2"][market],
                        "ML %": mlp["Resultado_1X2"][market],
                        "Diferencia pp": round(abs(mc["Resultado_1X2"][market] - mlp["Resultado_1X2"][market]), 1),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.markdown("### Mercados con linea real")
                gcol, ccol, tcol = st.columns(3)
                if lg is None:
                    gcol.info("Goles: sin linea real")
                else:
                    metric_market(
                        gcol,
                        f"Goles {lg}",
                        mc["Goles_Over_Under"],
                        f"Over {lg}", f"Under {lg}", f"Push {lg}"
                    )
                if lc is None:
                    ccol.info("Corners: sin linea real")
                else:
                    metric_market(
                        ccol,
                        f"Corners {lc}",
                        mc["Corners_Totales"],
                        f"Over {lc} Corners", f"Under {lc} Corners", f"Push {lc} Corners"
                    )
                if lt is None:
                    tcol.info("Tarjetas: sin linea real")
                else:
                    metric_market(
                        tcol,
                        f"Tarjetas {lt}",
                        mc["Tarjetas_Totales"],
                        f"Over {lt} Tarjetas", f"Under {lt} Tarjetas", f"Push {lt} Tarjetas"
                    )

                st.markdown("### Value betting")
                picks = evaluar_fixture(
                    local,
                    visita,
                    fixture_id,
                    df,
                    cuotas=cuotas,
                    ml=ml,
                    elo_map=elo_map,
                )
                if picks:
                    st.success(f"{len(picks)} oportunidad(es) superaron todos los filtros V2")
                    st.dataframe(pd.DataFrame(picks), use_container_width=True, hide_index=True)
                else:
                    st.info("NO BET — ninguna opcion supero probabilidad, edge, EV y acuerdo entre modelos.")

                with st.expander("Detalle tecnico"):
                    st.write("Cuotas observadas", cuotas)
                    st.write("Totales previstos por ML", mlp.get("Prediccion_Totales", {}))
                    st.write("Expectativas Monte Carlo", {
                        "goles_local": mc["Goles_Individuales"][local]["goles"],
                        "goles_visita": mc["Goles_Individuales"][visita]["goles"],
                        "corners_local": mc["Corners_Individuales"][local]["corners"],
                        "corners_visita": mc["Corners_Individuales"][visita]["corners"],
                    })

            except Exception as exc:
                st.error(f"NO BET / error de validacion: {exc}")

st.markdown("---")
st.subheader("Scanner de jornada V2")
st.caption("Entrena ML/ELO una sola vez y solo devuelve apuestas con cuota real, probabilidad no-vig, edge, EV y acuerdo entre modelos.")
if st.button("Escanear jornada completa V2"):
    with st.spinner("Escaneando proximos partidos..."):
        picks = escanear_jornada_actual(df)
    if picks:
        st.dataframe(pd.DataFrame(picks), use_container_width=True, hide_index=True)
    else:
        st.info("NO BET — no hay oportunidades que cumplan todos los filtros V2.")

st.markdown("---")
st.subheader("Ranking ELO actual")
st.dataframe(SistemaEloLigaMX().calcular_historico(df), use_container_width=True, hide_index=True)
