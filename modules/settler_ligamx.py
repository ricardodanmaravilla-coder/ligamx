import datetime
import os
import re
from urllib.parse import quote

import requests

from modules.sheets_ligamx import SPREADSHEET_ID, SHEET_NAME, _headers

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
FINAL_STATUSES = {"FT", "AET", "PEN"}
VOID_STATUSES = {"CANC", "ABD", "AWD", "WO"}


def _api_headers():
    return {"x-apisports-key": API_KEY} if API_KEY else {}


def _pending_rows():
    rng = quote(f"'{SHEET_NAME}'!A2:Y2000", safe="")
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{rng}"
    r = requests.get(url, headers=_headers(), timeout=20)
    r.raise_for_status()
    rows = r.json().get("values", [])
    out = []
    for sheet_row, row in enumerate(rows, start=2):
        padded = list(row) + [""] * (25 - len(row))
        if str(padded[20]).strip().lower() == "pending" and str(padded[3]).strip():
            out.append((sheet_row, padded[:25]))
    return out


def _fixture(fixture_id):
    if not API_KEY:
        raise RuntimeError("Falta API_SPORTS_KEY")
    r = requests.get(
        f"{BASE_URL}/fixtures",
        headers=_api_headers(),
        params={"id": int(fixture_id)},
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(f"API-Football fixtures: {payload['errors']}")
    response = payload.get("response", [])
    return response[0] if response else None


def _fixture_stats(fixture_id):
    r = requests.get(
        f"{BASE_URL}/fixtures/statistics",
        headers=_api_headers(),
        params={"fixture": int(fixture_id)},
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(f"API-Football statistics: {payload['errors']}")
    return payload.get("response", [])


def _num(value):
    if value is None or value == "":
        return 0.0
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    try:
        return float(value)
    except Exception:
        return 0.0


def _stat_totals(stats):
    total_corners = 0.0
    total_yellow = 0.0
    total_red = 0.0
    for team in stats or []:
        items = {
            str(x.get("type", "")): x.get("value")
            for x in team.get("statistics", [])
        }
        total_corners += _num(items.get("Corner Kicks"))
        total_yellow += _num(items.get("Yellow Cards"))
        total_red += _num(items.get("Red Cards"))
    return {
        "corners": total_corners,
        # Debe coincidir con feature_engineering.py: amarilla=1, roja=2.
        "cards": total_yellow + 2.0 * total_red,
        "yellow": total_yellow,
        "red": total_red,
    }


def _fulltime_goals(fx):
    score = fx.get("score", {}).get("fulltime", {}) or {}
    home = score.get("home")
    away = score.get("away")
    if home is None or away is None:
        goals = fx.get("goals", {}) or {}
        home, away = goals.get("home"), goals.get("away")
    if home is None or away is None:
        raise RuntimeError("Marcador final no disponible")
    return int(home), int(away)


def _total_selection(selection, observed):
    m = re.search(r"\b(Over|Under)\s+([0-9]+(?:\.[0-9]+)?)", str(selection), re.I)
    if not m:
        raise ValueError(f"Selección O/U no reconocida: {selection}")
    side = m.group(1).lower()
    line = float(m.group(2))
    if abs(float(observed) - line) < 1e-9:
        return "push", line
    won = observed > line if side == "over" else observed < line
    return ("won" if won else "lost"), line


def _settle_market(row, fx, stats_cache=None):
    market = str(row[7]).strip()
    selection = str(row[8]).strip()
    home_goals, away_goals = _fulltime_goals(fx)

    if market == "1X2":
        if home_goals > away_goals:
            actual = "Gana Local"
        elif home_goals < away_goals:
            actual = "Gana Visita"
        else:
            actual = "Empate"
        status = "won" if selection.lower() == actual.lower() else "lost"
        return status, f"{home_goals}-{away_goals} ({actual})"

    if market == "Goles":
        total = home_goals + away_goals
        status, line = _total_selection(selection, total)
        return status, f"Goles={total} | FT {home_goals}-{away_goals} | línea={line:g}"

    stats = stats_cache if stats_cache is not None else _fixture_stats(row[3])
    totals = _stat_totals(stats)
    if market == "Corners":
        observed = totals["corners"]
        status, line = _total_selection(selection, observed)
        return status, f"Corners={observed:g} | línea={line:g}"
    if market == "Tarjetas":
        observed = totals["cards"]
        status, line = _total_selection(selection, observed)
        return status, (
            f"Tarjetas={observed:g} (amarillas={totals['yellow']:g}, rojas={totals['red']:g}, roja×2) "
            f"| línea={line:g}"
        )
    raise ValueError(f"Mercado no soportado: {market}")


def _profit(status, odds, stake):
    odds = _num(odds)
    stake = _num(stake)
    if status == "won":
        units = max(0.0, odds - 1.0)
    elif status == "lost":
        units = -1.0
    else:  # push / void
        units = 0.0
    return round(units, 4), round(stake * units, 2)


def _update_result(sheet_row, status, result_value, profit_units, profit_mxn, settled_utc):
    rng = quote(f"'{SHEET_NAME}'!U{sheet_row}:Y{sheet_row}", safe="")
    url = (
        f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{rng}"
        "?valueInputOption=USER_ENTERED"
    )
    body = {
        "range": f"'{SHEET_NAME}'!U{sheet_row}:Y{sheet_row}",
        "majorDimension": "ROWS",
        "values": [[status, result_value, profit_units, profit_mxn, settled_utc]],
    }
    r = requests.put(url, headers=_headers(), json=body, timeout=20)
    if not r.ok:
        raise RuntimeError(f"Sheets {r.status_code}: {r.text[:500]}")


def liquidar_picks_pendientes():
    pending = _pending_rows()
    if not pending:
        return {"ok": True, "pending": 0, "settled": 0, "won": 0, "lost": 0, "push": 0, "void": 0, "skipped": 0}

    fixture_cache = {}
    stats_cache = {}
    summary = {"ok": True, "pending": len(pending), "settled": 0, "won": 0, "lost": 0, "push": 0, "void": 0, "skipped": 0, "errors": []}
    settled_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for sheet_row, row in pending:
        fixture_id = str(row[3]).strip()
        try:
            if fixture_id not in fixture_cache:
                fixture_cache[fixture_id] = _fixture(fixture_id)
            fx = fixture_cache[fixture_id]
            if not fx:
                summary["skipped"] += 1
                summary["errors"].append(f"{fixture_id}: fixture no encontrado")
                continue

            short = str(fx.get("fixture", {}).get("status", {}).get("short", ""))
            if short in VOID_STATUSES:
                status = "void"
                result_value = f"Partido {short}"
            elif short not in FINAL_STATUSES:
                summary["skipped"] += 1
                continue
            else:
                market = str(row[7]).strip()
                stats = None
                if market in {"Corners", "Tarjetas"}:
                    if fixture_id not in stats_cache:
                        stats_cache[fixture_id] = _fixture_stats(fixture_id)
                    stats = stats_cache[fixture_id]
                    if not stats:
                        summary["skipped"] += 1
                        summary["errors"].append(f"{fixture_id}: estadísticas finales no disponibles")
                        continue
                status, result_value = _settle_market(row, fx, stats)

            units, profit_mxn = _profit(status, row[9], row[17])
            _update_result(sheet_row, status, result_value, units, profit_mxn, settled_utc)
            summary["settled"] += 1
            summary[status] = summary.get(status, 0) + 1
        except Exception as exc:
            summary["skipped"] += 1
            summary["errors"].append(f"fila {sheet_row} fixture {fixture_id}: {type(exc).__name__}: {exc}")

    summary["errors"] = summary["errors"][:20]
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(liquidar_picks_pendientes(), ensure_ascii=False, indent=2))
