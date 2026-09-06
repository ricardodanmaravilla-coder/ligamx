import os
import requests

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Política de bookmaker:
# 1) Bet365 es la casa principal siempre que tenga el mercado COMPLETO.
# 2) Si Bet365 no ofrece ese mercado completo, se usa la siguiente casa disponible.
# 3) Nunca se mezclan lados de distintas casas dentro del mismo mercado.
PREFERRED_BOOKMAKER_NAMES = ["bet365", "1xbet", "pinnacle"]
PREFERRED_BOOKMAKER_IDS = [8, 6, 11, 1]


def _norm_market_name(name):
    return " ".join(str(name or "").strip().lower().replace("/", " ").split())


def _norm_bookmaker_name(name):
    return "".join(ch for ch in str(name or "").strip().lower() if ch.isalnum())


def _line_text(value):
    """Formato canónico para líneas: 10, 10.0 y 10.00 -> '10.0'; 9.5 -> '9.5'."""
    x = float(value)
    # Mantener al menos una decimal porque scanner/modelos trabajan con float.
    if abs(x - round(x)) < 1e-9:
        return f"{x:.1f}"
    s = f"{x:.2f}".rstrip("0")
    return s


def _candidate_from_book(book):
    """Extrae mercados de una casa sin mezclar lados entre bookmakers."""
    candidate = {
        "bookmaker_id": book.get("id"),
        "bookmaker_name": book.get("name", "Desconocido"),
    }

    for market in book.get("bets", []):
        raw_name = market.get("name", "")
        name = _norm_market_name(raw_name)
        vals = market.get("values", [])

        if name in {"match winner", "1x2", "winner"}:
            mp = {"Home": "1", "Draw": "X", "Away": "2"}
            for v in vals:
                key = mp.get(v.get("value"))
                if key:
                    try:
                        candidate[key] = float(v["odd"])
                    except (TypeError, ValueError, KeyError):
                        pass
        elif name in {"goals over under", "total goals", "goals over under fulltime"}:
            _extract_total(vals, candidate, "goles", "Goles")
        elif name in {"corners over under", "corners", "total corners", "corners total"}:
            _extract_total(vals, candidate, "corners", "Corners")
        elif name in {"cards over under", "cards", "total cards", "cards total"}:
            _extract_total(vals, candidate, "tarjetas", "Tarjetas")

    return candidate


def _preferred_rank(candidate):
    name = _norm_bookmaker_name(candidate.get("bookmaker_name"))
    for idx, preferred_name in enumerate(PREFERRED_BOOKMAKER_NAMES):
        if _norm_bookmaker_name(preferred_name) in name:
            return idx

    bid = candidate.get("bookmaker_id")
    if bid in PREFERRED_BOOKMAKER_IDS:
        return len(PREFERRED_BOOKMAKER_NAMES) + PREFERRED_BOOKMAKER_IDS.index(bid)
    return 999


def _best_1x2(candidates):
    complete = [c for c in candidates if all(c.get(k) for k in ("1", "X", "2"))]
    if not complete:
        return None
    return sorted(complete, key=lambda c: (_preferred_rank(c), -len(c)))[0]


def _best_total(candidates, kind, label):
    line_key = f"linea_{kind}_detectada"
    valid = []
    for c in candidates:
        line = c.get(line_key)
        if line is None:
            continue
        line = _line_text(line)
        over_key = f"Over {line} {label}"
        under_key = f"Under {line} {label}"
        if kind == "goles":
            over_key = f"Over {line}"
            under_key = f"Under {line}"
        if c.get(over_key) and c.get(under_key):
            balance = abs(float(c[over_key]) - float(c[under_key]))
            valid.append((_preferred_rank(c), balance, c))
    if not valid:
        return None
    valid.sort(key=lambda x: (x[0], x[1]))
    return valid[0][2]


def _merge_total(out, source, kind, label):
    if not source:
        return
    line_key = f"linea_{kind}_detectada"
    raw_line = source.get(line_key)
    if raw_line is None:
        return
    line = _line_text(raw_line)
    out[line_key] = line
    if kind == "goles":
        for key in (f"Over {line}", f"Under {line}"):
            if source.get(key) is not None:
                out[key] = source[key]
    else:
        for key in (f"Over {line} {label}", f"Under {line} {label}"):
            if source.get(key) is not None:
                out[key] = source[key]
    out[f"bookmaker_{kind}_id"] = source.get("bookmaker_id")
    out[f"bookmaker_{kind}_name"] = source.get("bookmaker_name", "Desconocido")


def obtener_cuotas_partido(fixture_id):
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

    books = []
    seen = set()
    for item in response:
        for book in item.get("bookmakers", []):
            marker = (book.get("id"), book.get("name"))
            if marker in seen:
                continue
            seen.add(marker)
            books.append(book)
    if not books:
        return {}

    candidates = [_candidate_from_book(book) for book in books]
    out = {"bookmakers_seen": len(candidates)}

    best_1x2 = _best_1x2(candidates)
    if best_1x2:
        out.update({
            "1": best_1x2["1"], "X": best_1x2["X"], "2": best_1x2["2"],
            "bookmaker_id": best_1x2.get("bookmaker_id"),
            "bookmaker_name": best_1x2.get("bookmaker_name", "Desconocido"),
        })

    _merge_total(out, _best_total(candidates, "goles", "Goles"), "goles", "Goles")
    _merge_total(out, _best_total(candidates, "corners", "Corners"), "corners", "Corners")
    _merge_total(out, _best_total(candidates, "tarjetas", "Tarjetas"), "tarjetas", "Tarjetas")

    useful = any(k in out for k in ("1", "linea_goles_detectada", "linea_corners_detectada", "linea_tarjetas_detectada"))
    return out if useful else {}


def _extract_total(vals, out, kind, label):
    pairs = {}
    for v in vals:
        s = str(v.get("value", "")).strip()
        low = s.lower()
        if not (low.startswith("over ") or low.startswith("under ")):
            continue
        side_raw, raw_line = s.split(" ", 1)
        side = "Over" if side_raw.lower() == "over" else "Under"
        try:
            odd = float(v.get("odd"))
            line = _line_text(raw_line)
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
