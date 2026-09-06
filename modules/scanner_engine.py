import os
from .feature_engineering import clean_history, normalize_team
from .elo_engine import SistemaEloLigaMX
from .ml_engine import PredictorML
from .montecarlo_sim import simular_partido_montecarlo
from .odds_engine import obtener_cuotas_partido, remove_vig_three_way, remove_vig_two_way

LIGA_MX_ID = 262
LIGA_MX_SEASON = 2026


def combinar_probabilidades(p_stat, p_ml, w_stat=0.55, w_ml=0.45):
    return w_stat * float(p_stat) + w_ml * float(p_ml)


def _disagreement_limit(p_stat, p_ml, base_limit=10.0):
    m = min(float(p_stat), float(p_ml))
    if m >= 60.0:
        return 15.0
    if m >= 55.0:
        return 12.0
    return float(base_limit)


def _market_probabilities_no_vig_1x2(cuotas):
    if not all(cuotas.get(k) for k in ("1", "X", "2")):
        return {}
    ph, pd, pa = remove_vig_three_way(cuotas["1"], cuotas["X"], cuotas["2"])
    return {"1": ph, "X": pd, "2": pa}


def _supported_total_line(line):
    try:
        q = round(float(line) * 4) / 4
        return abs(q - float(line)) < 1e-9
    except Exception:
        return False


def _total_spec(kind, line):
    line = float(line)
    if kind == "goles":
        return (
            "Goles O/U", "Goles_Over_Under",
            f"Over {line}", f"Under {line}", f"Push {line}",
            f"Over {line}", f"Under {line}",
        )
    if kind == "corners":
        return (
            "Corners O/U", "Corners_Totales",
            f"Over {line} Corners", f"Under {line} Corners", f"Push {line} Corners",
            f"Over {line} Corners", f"Under {line} Corners",
        )
    return (
        "Tarjetas O/U", "Tarjetas_Totales",
        f"Over {line} Tarjetas", f"Under {line} Tarjetas", f"Push {line} Tarjetas",
        f"Over {line} Tarjetas", f"Under {line} Tarjetas",
    )


def _evaluate_total_side(
    partido, market_name, side, p_stat, p_ml, push_stat, push_ml,
    odd, market_p, bookmaker, min_prob=60.0, min_edge=5.0,
    min_ev=4.0, max_disagreement=12.0,
    half_win_stat=0.0, half_win_ml=0.0,
    half_loss_stat=0.0, half_loss_ml=0.0,
    loss_stat=None, loss_ml=None,
):
    p_stat = float(p_stat)
    p_ml = float(p_ml)
    push_stat = float(push_stat or 0.0)
    push_ml = float(push_ml or 0.0)
    half_win_stat = float(half_win_stat or 0.0)
    half_win_ml = float(half_win_ml or 0.0)
    half_loss_stat = float(half_loss_stat or 0.0)
    half_loss_ml = float(half_loss_ml or 0.0)
    if loss_stat is None:
        loss_stat = max(0.0, 100.0 - p_stat - push_stat - half_win_stat - half_loss_stat)
    if loss_ml is None:
        loss_ml = max(0.0, 100.0 - p_ml - push_ml - half_win_ml - half_loss_ml)
    loss_stat = float(loss_stat)
    loss_ml = float(loss_ml)

    p_push = combinar_probabilidades(push_stat, push_ml)
    p_half_win = combinar_probabilidades(half_win_stat, half_win_ml)
    p_half_loss = combinar_probabilidades(half_loss_stat, half_loss_ml)
    p_loss = combinar_probabilidades(loss_stat, loss_ml)
    p_ensemble = combinar_probabilidades(p_stat, p_ml)

    score_stat = p_stat + 0.5 * half_win_stat
    score_ml = p_ml + 0.5 * half_win_ml
    disagreement = abs(score_stat - score_ml)
    allowed = _disagreement_limit(score_stat, score_ml, max_disagreement)

    win_units = p_ensemble + 0.5 * p_half_win
    loss_units = p_loss + 0.5 * p_half_loss
    if win_units <= 1e-9:
        fair_odds = 999.0
        p_fair = 0.0
    else:
        fair_odds = 1.0 + loss_units / win_units
        p_fair = 100.0 / fair_odds

    edge = p_fair - float(market_p)
    ev = (
        (p_ensemble / 100.0) * (float(odd) - 1.0)
        + (p_half_win / 100.0) * 0.5 * (float(odd) - 1.0)
        - (p_half_loss / 100.0) * 0.5
        - (p_loss / 100.0)
    ) * 100.0

    reasons = []
    if min(score_stat, score_ml) < 52.0:
        reasons.append("uno de los modelos <52%")
    if disagreement > allowed:
        reasons.append(f"desacuerdo {disagreement:.1f}>{allowed:.1f} pp")
    if p_fair < min_prob:
        reasons.append(f"prob. efectiva {p_fair:.1f}<{min_prob:.1f}%")
    if edge < min_edge:
        reasons.append(f"edge {edge:.1f}<{min_edge:.1f} pp")
    if ev < min_ev:
        reasons.append(f"EV {ev:.1f}<{min_ev:.1f}%")

    diag = {
        "Mercado": f"{market_name} {side}",
        "Bookmaker": bookmaker,
        "P_Estadistico": round(p_stat, 1),
        "P_ML": round(p_ml, 1),
        "P_HalfWin": round(p_half_win, 1),
        "P_Push": round(p_push, 1),
        "P_HalfLoss": round(p_half_loss, 1),
        "P_Loss": round(p_loss, 1),
        "P_Ensemble": round(p_ensemble, 1),
        "P_Condicional": round(p_fair, 1),
        "Cuota_Justa_Modelo": round(fair_odds, 3) if fair_odds < 999 else None,
        "Desacuerdo_pp": round(disagreement, 1),
        "Limite_Desacuerdo_pp": round(allowed, 1),
        "Cuota": round(float(odd), 2),
        "P_Mercado_NoVig": round(float(market_p), 1),
        "Edge_pp": round(edge, 1),
        "EV_pct": round(ev, 1),
    }
    if reasons:
        diag["Estado"] = "NO BET"
        diag["Motivo"] = "; ".join(reasons)
        return None, diag

    diag["Estado"] = "VALUE BET O/U — ASIAN V4"
    diag["Motivo"] = "Supera filtros con liquidación asiática correcta"
    pick = {"Partido": partido, **diag, "Veredicto": "VALUE BET O/U — ASIAN V4"}
    return pick, diag


def evaluar_fixture(local, visita, fixture_id, df_historico, cuotas=None, ml=None,
                    elo_map=None, min_prob=58.0, min_edge=4.0, min_ev=3.0,
                    max_disagreement=10.0, return_diagnostics=False):
    diagnostics = []
    df = clean_history(df_historico)
    local, visita = normalize_team(local), normalize_team(visita)
    partido = f"{local} vs {visita}"

    if elo_map is None:
        table = SistemaEloLigaMX().calcular_historico(df)
        elo_map = dict(zip(table.Equipo, table.ELO_Rating))
    if local not in elo_map or visita not in elo_map:
        diagnostics.append({"Mercado": "Todos", "Estado": "NO BET", "Motivo": "Equipo sin ELO histórico"})
        return ([], diagnostics) if return_diagnostics else []

    cuotas = cuotas or obtener_cuotas_partido(fixture_id)
    if not cuotas:
        diagnostics.append({"Mercado": "Todos", "Estado": "NO BET", "Motivo": "API-Football no devolvió cuotas"})
        return ([], diagnostics) if return_diagnostics else []

    if ml is None:
        ml = PredictorML()
        if not ml.entrenar(df):
            diagnostics.append({"Mercado": "Todos", "Estado": "NO BET", "Motivo": "ML no pudo entrenar"})
            return ([], diagnostics) if return_diagnostics else []

    lg = cuotas.get("linea_goles_detectada")
    lc = cuotas.get("linea_corners_detectada")
    lt = cuotas.get("linea_tarjetas_detectada")
    tech_g = float(lg) if lg is not None else 2.5
    tech_c = float(lc) if lc is not None else 9.5
    tech_t = float(lt) if lt is not None else 4.5

    try:
        mc = simular_partido_montecarlo(
            local, visita, df_historico=df,
            elo_local=elo_map[local], elo_visita=elo_map[visita],
            linea_goles=tech_g, linea_corners=tech_c, linea_tarjetas=tech_t,
        )
        mlp = ml.predecir_mercados_completos(
            df, local, visita,
            elo_local=elo_map[local], elo_visita=elo_map[visita],
            linea_goles=tech_g, linea_corners=tech_c, linea_tarjetas=tech_t,
        )
    except ValueError as exc:
        diagnostics.append({"Mercado": "Todos", "Estado": "NO BET", "Motivo": str(exc)})
        return ([], diagnostics) if return_diagnostics else []

    if mc.get("Contexto_Muestra", {}).get("fallback_stats"):
        diagnostics.append({
            "Mercado": "Contexto", "Estado": "INFO",
            "Motivo": "Muestra local/visitante corta: estadísticas encogidas hacia forma global y media de liga",
        })

    bookmaker_1x2 = cuotas.get("bookmaker_name", cuotas.get("bookmaker_id", "N/A"))
    out = []

    market_probs = _market_probabilities_no_vig_1x2(cuotas)
    if market_probs:
        markets = [("Gana Local", "1"), ("Empate", "X"), ("Gana Visita", "2")]
        for name, key in markets:
            p_stat = float(mc["Resultado_1X2"][name])
            p_ml = float(mlp["Resultado_1X2"][name])
            odd = cuotas.get(key)
            disagreement = abs(p_stat - p_ml)
            allowed = _disagreement_limit(p_stat, p_ml, max_disagreement)
            p_ensemble = combinar_probabilidades(p_stat, p_ml)
            market_p = market_probs[key]
            edge = p_ensemble - market_p
            ev = (p_ensemble / 100.0 * float(odd) - 1.0) * 100.0 if odd else -999
            reasons = []
            if not odd or float(odd) <= 1.01: reasons.append("sin cuota válida")
            if min(p_stat, p_ml) < 52.0: reasons.append("uno de los modelos <52%")
            if disagreement > allowed: reasons.append(f"desacuerdo {disagreement:.1f}>{allowed:.1f} pp")
            if p_ensemble < min_prob: reasons.append(f"ensemble {p_ensemble:.1f}<{min_prob:.1f}%")
            if edge < min_edge: reasons.append(f"edge {edge:.1f}<{min_edge:.1f} pp")
            if ev < min_ev: reasons.append(f"EV {ev:.1f}<{min_ev:.1f}%")
            diag = {
                "Mercado": name, "Bookmaker": bookmaker_1x2,
                "P_Estadistico": round(p_stat,1), "P_ML": round(p_ml,1),
                "P_Ensemble": round(p_ensemble,1), "Desacuerdo_pp": round(disagreement,1),
                "Limite_Desacuerdo_pp": round(allowed,1), "Cuota": round(float(odd),2) if odd else None,
                "P_Mercado_NoVig": round(market_p,1), "Edge_pp": round(edge,1), "EV_pct": round(ev,1)
            }
            if reasons:
                diag["Estado"]="NO BET"; diag["Motivo"]="; ".join(reasons)
            else:
                diag["Estado"]="VALUE BET 1X2"; diag["Motivo"]="Supera todos los filtros"
                out.append({**diag, "Partido": partido, "Veredicto":"VALUE BET 1X2"})
            diagnostics.append(diag)
    else:
        diagnostics.append({"Mercado":"1X2","Estado":"NO BET","Motivo":"No hay 1/X/2 completo para no-vig"})

    for kind, line in (("goles", lg), ("corners", lc), ("tarjetas", lt)):
        market_bookmaker = cuotas.get(
            f"bookmaker_{kind}_name",
            cuotas.get(f"bookmaker_{kind}_id", bookmaker_1x2),
        )
        if line is None:
            diagnostics.append({"Mercado":f"{kind.title()} O/U","Bookmaker":market_bookmaker,"Estado":"NO BET","Motivo":"Sin línea real en API-Football"})
            continue
        line = float(line)
        if not _supported_total_line(line):
            diagnostics.append({"Mercado":f"{kind.title()} O/U {line}","Bookmaker":market_bookmaker,"Estado":"NO BET","Motivo":"Línea fuera de incrementos asiáticos de 0.25"})
            continue
        market_name, section, over_model_key, under_model_key, push_key, over_odd_key, under_odd_key = _total_spec(kind, line)
        over_odd, under_odd = cuotas.get(over_odd_key), cuotas.get(under_odd_key)
        if not over_odd or not under_odd:
            diagnostics.append({"Mercado":f"{market_name} {line}","Bookmaker":market_bookmaker,"Estado":"NO BET","Motivo":"Faltan cuotas Over/Under completas"})
            continue
        market_over, market_under = remove_vig_two_way(over_odd, under_odd)
        mcsec, mlsec = mc.get(section, {}), mlp.get(section, {})
        if over_model_key not in mcsec or over_model_key not in mlsec:
            diagnostics.append({"Mercado":f"{market_name} {line}","Bookmaker":market_bookmaker,"Estado":"NO BET","Motivo":"Modelo no produjo la línea solicitada"})
            continue
        push_mc, push_ml = mcsec.get(push_key,0.0), mlsec.get(push_key,0.0)
        for side, mk, odd, mp in (
            (f"Over {line}", over_model_key, over_odd, market_over),
            (f"Under {line}", under_model_key, under_odd, market_under),
        ):
            pick, diag = _evaluate_total_side(
                partido, market_name, side,
                mcsec[mk], mlsec[mk], push_mc, push_ml,
                odd, mp, market_bookmaker,
                half_win_stat=mcsec.get(f"HalfWin {mk}", 0.0),
                half_win_ml=mlsec.get(f"HalfWin {mk}", 0.0),
                half_loss_stat=mcsec.get(f"HalfLoss {mk}", 0.0),
                half_loss_ml=mlsec.get(f"HalfLoss {mk}", 0.0),
                loss_stat=mcsec.get(f"Loss {mk}"),
                loss_ml=mlsec.get(f"Loss {mk}"),
            )
            diagnostics.append(diag)
            if pick:
                out.append(pick)

    out = sorted(out, key=lambda x: (x.get("EV_pct",0), x.get("Edge_pp",0)), reverse=True)
    return (out, diagnostics) if return_diagnostics else out


def escanear_jornada_actual(df_historico=None):
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
            params={"league": LIGA_MX_ID, "season": LIGA_MX_SEASON, "next": 15}, timeout=12,
        )
        payload = r.json() if r.status_code == 200 else {}
        fixtures = [] if payload.get("errors") else payload.get("response", [])
    except Exception:
        return []
    out=[]
    for f in fixtures:
        try:
            out.extend(evaluar_fixture(
                f["teams"]["home"]["name"], f["teams"]["away"]["name"],
                f["fixture"]["id"], df, ml=ml, elo_map=elo_map
            ))
        except Exception:
            continue
    return sorted(out, key=lambda x:(x.get("EV_pct",0),x.get("Edge_pp",0)), reverse=True)
