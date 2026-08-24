import os
import requests

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}


def obtener_cuotas_partido(fixture_id):
    """Devuelve exclusivamente mercados y lineas observados en API-Football."""
    if not fixture_id or not API_KEY:
        return {}

    url = f"{BASE_URL}/odds"
    out = {}

    for bookmaker_id in [8, 6, 11, 1]:
        try:
            r = requests.get(
                url,
                headers=HEADERS,
                params={"fixture": fixture_id, "bookmaker": bookmaker_id},
                timeout=10,
            )
            if r.status_code != 200:
                continue
            data = r.json().get("response", [])
            if not data:
                continue
            books = data[0].get("bookmakers", [])
            if not books:
                continue

            candidate = {}
            candidate["bookmaker_id"] = bookmaker_id
            for market in books[0].get("bets", []):
                name = market.get("name", "")
                vals = market.get("values", [])

                if name == "Match Winner":
                    for v in vals:
                        mp = {"Home": "1", "Draw": "X", "Away": "2"}
                        if v.get("value") in mp:
                            try:
                                candidate[mp[v["value"]]] = float(v["odd"])
                            except (TypeError, ValueError):
                                pass
                elif name == "Goals Over/Under":
                    _extract_total(vals, candidate, "goles", "Goles")
                elif name in ["Corners Over Under", "Corners", "Total Corners"]:
                    _extract_total(vals, candidate, "corners", "Corners")
                elif name in ["Cards Over/Under", "Cards", "Total Cards"]:
                    _extract_total(vals, candidate, "tarjetas", "Tarjetas")

            # No mezclar cuotas de distintas casas: conservamos un snapshot coherente.
            if len(candidate) > 1:
                out = candidate
                if all(k in out for k in ("1", "X", "2")):
                    break
        except Exception:
            continue

    return out


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

    complete = [
        (line, p) for line, p in pairs.items()
        if "Over" in p and "Under" in p
    ]
    if not complete:
        return

    # Elegimos la linea principal aproximando el punto de cuotas equilibradas.
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
    implied = np = [
        1.0 / float(odd_home),
        1.0 / float(odd_draw),
        1.0 / float(odd_away),
    ]
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
    """Compatibilidad con la interfaz manual. Solo evalua cuotas presentes."""
    import pandas as pd

    cuotas = cuotas_personalizadas or obtener_cuotas_partido(fixture_id) or {}
    rows = []
    markets = []
    r1 = resultados_montecarlo.get("Resultado_1X2", {})

    for name, key in [("Gana Local", "1"), ("Empate", "X"), ("Gana Visita", "2")]:
        markets.append((name, r1.get(name), key))

    for section in ["Goles_Over_Under", "Corners_Totales", "Tarjetas_Totales"]:
        for key, p in resultados_montecarlo.get(section, {}).items():
            if key.startswith(("Over ", "Under ")):
                markets.append((key, p, key))

    no_vig_1x2 = None
    if all(cuotas.get(k) for k in ("1", "X", "2")):
        no_vig_1x2 = dict(zip(
            ("1", "X", "2"),
            remove_vig_three_way(cuotas["1"], cuotas["X"], cuotas["2"]),
        ))

    for name, p, key in markets:
        odd = cuotas.get(key)
        if not odd or p is None:
            rows.append([name, f"{p or 0}%", "Sin Cuota", "N/A", "N/A", "0%", "NO BET"])
            continue

        market_p = no_vig_1x2.get(key) if no_vig_1x2 and key in no_vig_1x2 else None
        if key.startswith("Over "):
            opp = "Under " + key[5:]
            if cuotas.get(opp):
                market_p = remove_vig_two_way(odd, cuotas[opp])[0]
        elif key.startswith("Under "):
            opp = "Over " + key[6:]
            if cuotas.get(opp):
                market_p = remove_vig_two_way(odd, cuotas[opp])[0]

        ev = evaluar_mercado(float(p), float(odd), market_p)
        edge_ok = ev["edge_pp"] is None or ev["edge_pp"] >= 3.0
        verdict = "VALUE BET" if ev["ev_pct"] >= 3.0 and float(p) >= 55.0 and edge_ok else "NO BET"
        stake = min(ev["kelly_pct"] * 0.25, 2.0) if verdict == "VALUE BET" else 0.0
        rows.append([
            name,
            f"{float(p):.1f}%",
            float(odd),
            f"{ev['ev_pct']:.1f}%",
            "Controlado" if verdict == "VALUE BET" else "Alto",
            f"{stake:.1f}%",
            verdict,
        ])

    return pd.DataFrame(
        rows,
        columns=["Mercado", "Prob. Modelo", "Cuota", "EV", "Riesgo", "Stake (Bankroll)", "Veredicto"],
    )
