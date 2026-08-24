import os
from .feature_engineering import clean_history, normalize_team
from .elo_engine import SistemaEloLigaMX
from .ml_engine import PredictorML
from .montecarlo_sim import simular_partido_montecarlo
from .odds_engine import obtener_cuotas_partido, remove_vig_three_way

LIGA_MX_ID = 262
LIGA_MX_SEASON = 2026


def combinar_probabilidades(p_stat, p_ml, w_stat=0.55, w_ml=0.45):
    """Ensemble provisional; no interpreta los modelos como evidencia independiente."""
    return w_stat * float(p_stat) + w_ml * float(p_ml)


def _market_probabilities_no_vig_1x2(cuotas):
    """Probabilidades 1X2 sin margen. Requiere las tres cuotas reales."""
    if not all(cuotas.get(k) for k in ("1", "X", "2")):
        return {}
    ph, pd, pa = remove_vig_three_way(cuotas["1"], cuotas["X"], cuotas["2"])
    return {"1": ph, "X": pd, "2": pa}


def evaluar_fixture(
    local,
    visita,
    fixture_id,
    df_historico,
    cuotas=None,
    ml=None,
    elo_map=None,
    min_prob=58.0,
    min_edge=4.0,
    min_ev=3.0,
    max_disagreement=10.0,
):
    """Scanner V2 conservador; solo recomienda 1X2 validado OOS."""
    df = clean_history(df_historico)
    local, visita = normalize_team(local), normalize_team(visita)

    if elo_map is None:
        table = SistemaEloLigaMX().calcular_historico(df)
        elo_map = dict(zip(table.Equipo, table.ELO_Rating))
    if local not in elo_map or visita not in elo_map:
        return []

    cuotas = cuotas or obtener_cuotas_partido(fixture_id)
    if not cuotas:
        return []

    market_probs = _market_probabilities_no_vig_1x2(cuotas)
    if not market_probs:
        return []

    if ml is None:
        ml = PredictorML()
        if not ml.entrenar(df):
            return []

    mc = simular_partido_montecarlo(
        local,
        visita,
        df_historico=df,
        elo_local=elo_map[local],
        elo_visita=elo_map[visita],
        linea_goles=2.5,
        linea_corners=9.5,
        linea_tarjetas=4.5,
    )
    mlp = ml.predecir_mercados_completos(
        df,
        local,
        visita,
        elo_local=elo_map[local],
        elo_visita=elo_map[visita],
        linea_goles=2.5,
        linea_corners=9.5,
        linea_tarjetas=4.5,
    )

    markets = [
        ("Gana Local", "1", mc["Resultado_1X2"]["Gana Local"], mlp["Resultado_1X2"]["Gana Local"]),
        ("Empate", "X", mc["Resultado_1X2"]["Empate"], mlp["Resultado_1X2"]["Empate"]),
        ("Gana Visita", "2", mc["Resultado_1X2"]["Gana Visita"], mlp["Resultado_1X2"]["Gana Visita"]),
    ]

    out = []
    for name, key, p_stat, p_ml in markets:
        odd = cuotas.get(key)
        if not odd or float(odd) <= 1.01:
            continue

        p_stat = float(p_stat)
        p_ml = float(p_ml)
        disagreement = abs(p_stat - p_ml)
        if disagreement > max_disagreement:
            continue
        if min(p_stat, p_ml) < 52.0:
            continue

        p_ensemble = combinar_probabilidades(p_stat, p_ml)
        market_p = market_probs[key]
        edge = p_ensemble - market_p
        ev = (p_ensemble / 100.0 * float(odd) - 1.0) * 100.0

        if p_ensemble >= min_prob and edge >= min_edge and ev >= min_ev:
            out.append({
                "Partido": f"{local} vs {visita}",
                "Mercado": name,
                "P_Estadistico": round(p_stat, 1),
                "P_ML": round(p_ml, 1),
                "Desacuerdo_pp": round(disagreement, 1),
                "P_Ensemble": round(p_ensemble, 1),
                "Cuota": round(float(odd), 2),
                "P_Mercado_NoVig": round(market_p, 1),
                "Edge_pp": round(edge, 1),
                "EV_pct": round(ev, 1),
                "Veredicto": "VALUE BET 1X2",
            })

    return sorted(out, key=lambda x: (x["EV_pct"], x["Edge_pp"]), reverse=True)


def escanear_jornada_actual(df_historico=None):
    """Escanea próximos fixtures; entrena ML/ELO una sola vez por ejecución."""
    import requests
    from .stats_engine import cargar_datos

    if df_historico is None:
        df_historico = cargar_datos()
    df = clean_history(df_historico)

    api_key = os.environ.get("API_SPORTS_KEY")
    if not api_key:
        return []

    ml = PredictorML()
    if not ml.entrenar(df):
        return []
    table = SistemaEloLigaMX().calcular_historico(df)
    elo_map = dict(zip(table.Equipo, table.ELO_Rating))

    try:
        r = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers={"x-apisports-key": api_key},
            params={"league": LIGA_MX_ID, "season": LIGA_MX_SEASON, "next": 15},
            timeout=12,
        )
        if r.status_code != 200:
            return []
        payload = r.json()
        if payload.get("errors"):
            return []
        fixtures = payload.get("response", [])
    except Exception:
        return []

    out = []
    for f in fixtures:
        try:
            local = f["teams"]["home"]["name"]
            visita = f["teams"]["away"]["name"]
            fid = f["fixture"]["id"]
            out.extend(evaluar_fixture(local, visita, fid, df, ml=ml, elo_map=elo_map))
        except Exception:
            continue

    return sorted(out, key=lambda x: (x.get("EV_pct", 0), x.get("Edge_pp", 0)), reverse=True)
