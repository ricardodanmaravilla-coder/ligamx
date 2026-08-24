import os
import requests

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Preferencia histórica, pero ya NO limita la búsqueda.
PREFERRED_BOOKMAKER_IDS = [8, 6, 11, 1]


def _candidate_from_book(book):
    """Extrae mercados de una sola casa; nunca mezcla cuotas entre bookmakers."""
    candidate = {
        "bookmaker_id": book.get("id"),
        "bookmaker_name": book.get("name", "Desconocido"),
    }

    for market in book.get("bets", []):
        name = market.get("name", "")
        vals = market.get("values", [])

        if name == "Match Winner":
            mp = {"Home": "1", "Draw": "X", "Away": "2"}
            for v in vals:
                key = mp.get(v.get("value"))
                if key:
                    try:
                        candidate[key] = float(v["odd"])
                    except (TypeError, ValueError, KeyError):
                        pass
        elif name == "Goals Over/Under":
            _extract_total(vals, candidate, "goles", "Goles")
        elif name in ["Corners Over Under", "Corners", "Total Corners"]:
            _extract_total(vals, candidate, "corners", "Corners")
        elif name in ["Cards Over/Under", "Cards", "Total Cards"]:
            _extract_total(vals, candidate, "tarjetas", "Tarjetas")

    return candidate


def obtener_cuotas_partido(fixture_id):
    """
    Devuelve un snapshot coherente de UNA sola casa.

    V2 anterior consultaba únicamente cuatro bookmaker IDs. Eso podía devolver
    cuotas vacías aunque API-Football tuviera mercado 1X2 en otra casa. Ahora
    consultamos todas las casas del fixture y priorizamos una con 1/X/2 completo.
    """
    if not fixture_id or not API_KEY:
        return {}

    try:
        r = requests.get(
            f"{BASE_URL}/odds",
            headers=HEADERS,
            params={"fixture": fixture_id},
            timeout=12,
        )
    except Exception:
        return {}

    if r.status_code != 200:
        return {}

    try:
        payload = r.json()
    except Exception:
        return {}

    if payload.get("errors"):
        return {}

    response = payload.get("response", [])
    if not response:
        return {}

    books = response[0].get("bookmakers", [])
    if not books:
        return {}

    candidates = [_candidate_from_book(book) for book in books]

    # Primero exigimos 1X2 completo para que el scanner pueda calcular no-vig.
    complete_1x2 = [c for c in candidates if all(c.get(k) for k in ("1", "X", "2"))]
    if complete_1x2:
        def rank(c):
            bid = c.get("bookmaker_id")
            preferred = PREFERRED_BOOKMAKER_IDS.index(bid) if bid in PREFERRED_BOOKMAKER_IDS else 999
            # Desempate: preferimos el snapshot que además tenga más mercados útiles.
            richness = len(c)
            return (preferred, -richness)

        return sorted(complete_1x2, key=rank)[0]

    # Si no existe 1X2 completo, devolvemos el snapshot más rico SOLO para diagnóstico.
    # El scanner seguirá respondiendo NO BET porque no puede quitar el vig 1X2.
    candidates = [c for c in candidates if len(c) > 2]
    return max(candidates, key=len) if candidates else {}


def _extract_total(vals, out, kind, label):
    pairs = {}
    for v in vals:
        s = str(v.get("value", ""))
        if not (s.startswith("Over ") or s.startswith("Under ")):
            continue
        side, line = s.split(" ", 1)
        try:
            odd = float(v.get("odd"))
            float(line)
        except (TypeError, ValueError):
            continue
        pairs.setdefault(line, {})[side] = odd

    complete = [(line, p) for line, p in pairs.items() if "Over" in p and "Under" in p]
    if not complete:
        return

    line, p = min(complete, key=lambda x: abs(x[1]["Over"] - x[1]["Under"]))
    out[f"linea_{kind}_detectada"] = line
    out[f"Over {line} {label}"] = p["Over"]
    out[f"Under {line} {label}"] = p["Under"]
    if kind == "goles":
        out[f"Over {line}"] = p["Over"]
        out[f"Under {line}"] = p["Under"]


def remove_vig_two_way(odd_a, odd_b):
    ia, ib = 1.0 / float(odd_a), 1.0 / float(odd_b)
    s = ia + ib
    return ia / s * 100.0, ib / s * 100.0


def remove_vig_three_way(odd_home, odd_draw, odd_away):
    implied = [1.0 / float(odd_home), 1.0 / float(odd_draw), 1.0 / float(odd_away)]
    s = sum(implied)
    return tuple(v / s * 100.0 for v in implied)


def evaluar_mercado(prob_pct, cuota, market_prob_pct=None):
    if not cuota or float(cuota) <= 1.0:
        return None
    cuota = float(cuota)
    p = float(prob_pct) / 100.0
    ev = (p * cuota - 1.0) * 100.0
    edge = None if market_prob_pct is None else float(prob_pct) - float(market_prob_pct)
    b = cuota - 1.0
    kelly = max(0.0, ((b * p) - (1.0 - p)) / b) * 100.0 if b > 0 else 0.0
    return {
        "cuota": cuota,
        "prob_modelo": float(prob_pct),
        "prob_mercado_no_vig": market_prob_pct,
        "edge_pp": edge,
        "ev_pct": ev,
        "kelly_pct": kelly,
    }


def analizar_apuestas(resultados_montecarlo, fixture_id, cuotas_personalizadas=None):
    """Compatibilidad V1; solo 1X2 puede generar VALUE BET en V2."""
    import pandas as pd

    cuotas = cuotas_personalizadas or obtener_cuotas_partido(fixture_id) or {}
    rows = []
    markets = []
    r1 = resultados_montecarlo.get("Resultado_1X2", {})

    for name, key in [("Gana Local", "1"), ("Empate", "X"), ("Gana Visita", "2")]:
        markets.append((name, r1.get(name), key, True))

    for section in ["Goles_Over_Under", "Corners_Totales", "Tarjetas_Totales"]:
        for key, p in resultados_montecarlo.get(section, {}).items():
            if key.startswith(("Over ", "Under ")):
                markets.append((key, p, key, False))

    no_vig_1x2 = None
    if all(cuotas.get(k) for k in ("1", "X", "2")):
        no_vig_1x2 = dict(zip(("1", "X", "2"), remove_vig_three_way(cuotas["1"], cuotas["X"], cuotas["2"])))

    for name, p, key, enabled_for_bet in markets:
        odd = cuotas.get(key)
        if not odd or p is None:
            rows.append([name, f"{p or 0}%", "Sin Cuota", "N/A", "N/A", "0%", "NO BET"])
            continue

        if not enabled_for_bet:
            rows.append([name, f"{float(p):.1f}%", float(odd), "Experimental", "No validado OOS", "0%", "NO BET — EXPERIMENTAL"])
            continue

        market_p = no_vig_1x2.get(key) if no_vig_1x2 else None
        if market_p is None:
            rows.append([name, f"{float(p):.1f}%", float(odd), "N/A", "Sin no-vig 1X2", "0%", "NO BET"])
            continue

        ev = evaluar_mercado(float(p), float(odd), market_p)
        edge_ok = ev["edge_pp"] is not None and ev["edge_pp"] >= 4.0
        verdict = "VALUE BET 1X2" if ev["ev_pct"] >= 3.0 and float(p) >= 58.0 and edge_ok else "NO BET"
        stake = min(ev["kelly_pct"] * 0.25, 1.5) if verdict == "VALUE BET 1X2" else 0.0
        rows.append([name, f"{float(p):.1f}%", float(odd), f"{ev['ev_pct']:.1f}%", "Controlado" if verdict == "VALUE BET 1X2" else "Alto", f"{stake:.1f}%", verdict])

    return pd.DataFrame(rows, columns=["Mercado", "Prob. Modelo", "Cuota", "EV", "Riesgo", "Stake (Bankroll)", "Veredicto"])
