"""Fuentes externas gratuitas: StrikerAPI y Football-Data (Liga MX).

Las cuotas solo se aceptan cuando se verifica fecha y ambos equipos.
Nunca se convierten odds historicas en odds prepartido actuales.
"""
import os
import re
import unicodedata
from datetime import datetime, timezone

import pandas as pd
import requests

STRIKER_BASE = "https://mcp.strikerapi.com/v1"
FOOTBALL_DATA_MEX_CSV = "https://www.football-data.co.uk/new/MEX.csv"
TIMEOUT = 15

ALIASES = {
    "america": "america", "club america": "america",
    "chivas": "guadalajara", "guadalajara chivas": "guadalajara",
    "pumas unam": "pumas", "unam pumas": "pumas", "u n a m pumas": "pumas",
    "tigres uanl": "tigres", "uanl tigres": "tigres",
    "atletico san luis": "atletico san luis",
    "fc juarez": "juarez", "club tijuana": "tijuana",
    "cf pachuca": "pachuca", "club queretaro": "queretaro",
    "leon": "leon", "mazatlan fc": "mazatlan",
}
def team_key(name):
    raw = unicodedata.normalize("NFKD", str(name or ""))
    raw = "".join(c for c in raw if not unicodedata.combining(c))
    raw = re.sub(r"[^a-z0-9 ]", " ", raw.lower())
    raw = " ".join(raw.split())
    return ALIASES.get(raw, raw)

def _post_striker(tool, body):
    key = os.getenv("STRIKER_API_KEY")
    if not key:
        return {}
    try:
        r = requests.post(
            f"{STRIKER_BASE}/{tool}", headers={"x-api-key": key},
            json=dict(body, league="ligamx"), timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else {}
    except (requests.RequestException, ValueError):
        return {}

def _american_to_decimal(value):
    try:
        val = float(value)
        if val == 0:
            return None
        return round(1 + (val / 100 if val > 0 else 100 / abs(val)), 5)
    except (TypeError, ValueError, ZeroDivisionError):
        return None

def cuotas_striker(local, visita, fecha):
    """Recupera snapshot por game_id verificado; no usa fixture_id de API-Football."""
    if not os.getenv("STRIKER_API_KEY") or not local or not visita or not fecha:
        return {}
    date = str(fecha)[:10]
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {}
    games = _post_striker("list_games", {"date": date, "limit": 100}).get("games", [])
    matches = [
        g for g in games
        if g.get("date") == date
        and team_key((g.get("home") or {}).get("name")) == team_key(local)
        and team_key((g.get("away") or {}).get("name")) == team_key(visita)
        and g.get("game_id")
    ]
    if len(matches) != 1:
        return {}  # Sin identificacion inequivoca => NO BET.
    response = _post_striker("get_odds", {"game_id": str(matches[0]["game_id"])})
    priced = [g for g in response.get("games", []) if str(g.get("game_id")) == str(matches[0]["game_id"])]
    if len(priced) != 1:
        return {}
    game = priced[0]
    candidates = []
    for book in game.get("books", []):
        # close es el precio disponible en el feed; no se presupone que sea live.
        snap = {"bookmaker_name": book.get("provider", "StrikerAPI"), "source": "StrikerAPI",
                "captured_at": book.get("captured_at"), "game_id": game["game_id"]}
        for key, source in (("1", "home_ml_close"), ("X", "draw_ml_close"), ("2", "away_ml_close")):
            price = _american_to_decimal(book.get(source))
            if price and price > 1:
                snap[key] = price
        line = book.get("total_close")
        over, under = _american_to_decimal(book.get("over_odds")), _american_to_decimal(book.get("under_odds"))
        if line is not None and over and under:
            try:
                line = str(float(line)).rstrip("0").rstrip(".")
                snap["linea_goles_detectada"] = line
                snap[f"Over {line}"] = over
                snap[f"Under {line}"] = under
                snap[f"Over {line} Goles"] = over
                snap[f"Under {line} Goles"] = under
            except (ValueError, TypeError):
                pass
        if all(snap.get(k) for k in ("1", "X", "2")):
            candidates.append(snap)
    # No mezclar casas; la mejor cobertura completa en una sola casa.
    return max(candidates, key=lambda x: len(x)) if candidates else {}

def descargar_historico_football_data():
    """DataFrame separado para backtesting; nunca reemplaza el historico de features."""
    r = requests.get(FOOTBALL_DATA_MEX_CSV, timeout=TIMEOUT)
    r.raise_for_status()
    from io import BytesIO
    df = pd.read_csv(BytesIO(r.content))
    if df.empty:
        raise ValueError("Football-Data devolvio CSV vacio")
    return df.assign(fuente="Football-Data", descargado_utc=datetime.now(timezone.utc).isoformat())
