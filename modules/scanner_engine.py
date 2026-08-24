import os
import pandas as pd
from .feature_engineering import clean_history, normalize_team
from .elo_engine import SistemaEloLigaMX
from .ml_engine import PredictorML
from .montecarlo_sim import simular_partido_montecarlo
from .odds_engine import obtener_cuotas_partido, remove_vig_two_way, remove_vig_three_way


def combinar_probabilidades(p_stat, p_ml, w_stat=0.55, w_ml=0.45):
    """Ensemble provisional. Los pesos se ajustaran con backtest walk-forward."""
    return w_stat * float(p_stat) + w_ml * float(p_ml)


def _market_probabilities_no_vig(cuotas):
    out = {}
    if all(cuotas.get(k) for k in ("1", "X", "2")):
        ph, pd, pa = remove_vig_three_way(cuotas["1"], cuotas["X"], cuotas["2"])
        out.update({"1": ph, "X": pd, "2": pa})

    for key, odd in list(cuotas.items()):
        if not isinstance(key, str) or not key.startswith(("Over ", "Under ")):
            continue
        if key.startswith("Over "):
            opp = "Under " + key[5:]
        else:
            opp = "Over " + key[6:]
        if cuotas.get(opp):
            out[key] = remove_vig_two_way(odd, cuotas[opp])[0]
    return out


def evaluar_fixture(
    local,
    visita,
    fixture_id,
    df_historico,
    cuotas=None,
    ml=None,
    elo_map=None,
    min_prob=55.0,
    min_edge=3.0,
    min_ev=3.0,
    max_disagreement=12.0,
):
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

    if ml is None:
        ml = PredictorML()
        if not ml.entrenar(df):
            return []

    lg = cuotas.get("linea_goles_detectada")
    lc = cuotas.get("linea_corners_detectada")
    lt = cuotas.get("linea_tarjetas_detectada")

    # Para 1X2 el simulador necesita lineas tecnicas, pero esas lineas nunca se
    # evaluan si no fueron observadas en cuotas.
    mc = simular_partido_montecarlo(
        local,
        visita,
        df_historico=df,
        elo_local=elo_map[local],
        elo_visita=elo_map[visita],
        linea_goles=float(lg) if lg else 2.5,
        linea_corners=float(lc) if lc else 9.5,
        linea_tarjetas=float(lt) if lt else 4.5,
    )
    mlp = ml.predecir_mercados_completos(
        df,
        local,
        visita,
        elo_local=elo_map[local],
        elo_visita=elo_map[visita],
        linea_goles=float(lg) if lg else 2.5,
        linea_corners=float(lc) if lc else 9.5,
        linea_tarjetas=float(lt) if lt else 4.5,
    )

    markets = [
        ("Gana Local", "1", mc["Resultado_1X2"]["Gana Local"], mlp["Resultado_1X2"]["Gana Local"]),
        ("Empate", "X", mc["Resultado_1X2"]["Empate"], mlp["Resultado_1X2"]["Empate"]),
        ("Gana Visita", "2", mc["Resultado_1X2"]["Gana Visita"], mlp["Resultado_1X2"]["Gana Visita"]),
    ]

    if lg:
        line = float(lg)
        markets += [
            (f"Over {lg} Goles", f"Over {lg}", mc["Goles_Over_Under"][f"Over {line}"], mlp["Goles_Over_Under"][f"Over {line}"]),
            (f"Under {lg} Goles", f"Under {lg}", mc["Goles_Over_Under"][f"Under {line}"], mlp["Goles_Over_Under"][f"Under {line}"]),
        ]
    if lc:
        line = float(lc)
        markets += [
            (f"Over {lc} Corners", f"Over {lc} Corners", mc["Corners_Totales"][f"Over {line} Corners"], mlp["Corners_Totales"][f"Over {line} Corners"]),
            (f"Under {lc} Corners", f"Under {lc} Corners", mc["Corners_Totales"][f"Under {line} Corners"], mlp["Corners_Totales"][f"Under {line} Corners"]),
        ]
    if lt:
        line = float(lt)
        markets += [
            (f"Over {lt} Tarjetas", f"Over {lt} Tarjetas", mc["Tarjetas_Totales"][f"Over {line} Tarjetas"], mlp["Tarjetas_Totales"][f"Over {line} Tarjetas"]),
            (f"Under {lt} Tarjetas", f"Under {lt} Tarjetas", mc["Tarjetas_Totales"][f"Under {line} Tarjetas"], mlp["Tarjetas_Totales"][f"Under {line} Tarjetas"]),
        ]

    market_probs = _market_probabilities_no_vig(cuotas)
    out = []

    for name, key, p_stat, p_ml in markets:
        odd = cuotas.get(key)
        if not odd or float(odd) <= 1.01:
            continue

        disagreement = abs(float(p_stat) - float(p_ml))
        if disagreement > max_disagreement:
            continue

        p_ensemble = combinar_probabilidades(p_stat, p_ml)
        market_p = market_probs.get(key)
        if market_p is None:
            # Sin todos los lados no estimamos edge limpio de margen.
            continue

        edge = p_ensemble - market_p
        ev = (p_ensemble / 100.0 * float(odd) - 1.0) * 100.0

        if p_ensemble >= min_prob and edge >= min_edge and ev >= min_ev:
            out.append({
                "Partido": f"{local} vs {visita}",
                "Mercado": name,
                "P_Estadistico": round(float(p_stat), 1),
                "P_ML": round(float(p_ml), 1),
                "Desacuerdo_pp": round(disagreement, 1),
                "P_Ensemble": round(p_ensemble, 1),
                "Cuota": round(float(odd), 2),
                "P_Mercado": round(market_p, 1),
                "Edge_pp": round(edge, 1),
                "EV_pct": round(ev, 1),
                "Veredicto": "VALUE BET",
            })

    return sorted(out, key=lambda x: (x["EV_pct"], x["Edge_pp"]), reverse=True)


def escanear_jornada_actual(df_historico=None):
    """Escanea proximos fixtures. Entrena ML/ELO una sola vez por ejecucion."""
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
            params={"league": 262, "next": 15},
            timeout=12,
        )
        if r.status_code != 200:
            return []
        fixtures = r.json().get("response", [])
    except Exception:
        return []

    out = []
    for f in fixtures:
        try:
            local = f["teams"]["home"]["name"]
            visita = f["teams"]["away"]["name"]
            fid = f["fixture"]["id"]
            out.extend(evaluar_fixture(
                local,
                visita,
                fid,
                df,
                ml=ml,
                elo_map=elo_map,
            ))
        except Exception:
            continue

    return sorted(out, key=lambda x: (x.get("EV_pct", 0), x.get("Edge_pp", 0)), reverse=True)
