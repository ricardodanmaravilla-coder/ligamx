import datetime
import os
from urllib.parse import quote

import google.auth
from google.auth.transport.requests import Request
import requests

SPREADSHEET_ID = os.environ.get("LIGAMX_SHEET_ID", "1VsB21QUsQL5EyXu7Sek5WVeNVznECiTuoIMMB4JXno4")
SHEET_NAME = os.environ.get("LIGAMX_SHEET_TAB", "LigaMX_Picks")
MODEL_VERSION = "ligamx-v4-cloudrun"
BANKROLL_MXN = float(os.environ.get("LIGAMX_BANKROLL_MXN", "5000"))

HEADERS = [
    "record_key", "snapshot_utc", "game_date", "fixture_id", "league",
    "away", "home", "market", "selection", "odds", "prob_ml", "prob_mc",
    "prob_combined", "disagreement_pp", "ev_pct", "kelly_pct", "bankroll_mxn",
    "stake_mxn", "verdict", "model_version", "result_status", "result_value",
    "profit_units", "profit_mxn", "settled_utc",
]


def _access_token():
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    if not credentials.valid:
        credentials.refresh(Request())
    return credentials.token


def _headers():
    return {
        "Authorization": f"Bearer {_access_token()}",
        "Content-Type": "application/json",
    }


def _existing_keys():
    rng = quote(f"'{SHEET_NAME}'!A2:A2000", safe="")
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{rng}"
    r = requests.get(url, headers=_headers(), timeout=15)
    r.raise_for_status()
    values = r.json().get("values", [])
    return {str(row[0]) for row in values if row and row[0]}


def _market_selection(raw_market):
    text = str(raw_market or "").strip()
    mappings = [
        ("Goles O/U", "Goles"),
        ("Corners O/U", "Corners"),
        ("Tarjetas O/U", "Tarjetas"),
    ]
    for prefix, market in mappings:
        if text.startswith(prefix):
            return market, text[len(prefix):].strip()
    return "1X2", text


def _kelly_fraction(prob_pct, odds):
    try:
        p = max(0.0, min(1.0, float(prob_pct) / 100.0))
        o = float(odds)
        b = o - 1.0
        if b <= 0:
            return 0.0
        full = (b * p - (1.0 - p)) / b
        # Mismo enfoque conservador usado en el seguimiento de Fut Europa: 20% Kelly.
        return max(0.0, full) * 0.20
    except Exception:
        return 0.0


def _pick_row(pick, now_utc):
    fixture_id = str(pick.get("fixture_id", ""))
    game_date = str(pick.get("game_date", pick.get("Fecha", "")))[:10]
    home = str(pick.get("home", pick.get("Local", "")))
    away = str(pick.get("away", pick.get("Visita", "")))
    market, selection = _market_selection(pick.get("Mercado"))
    odds = float(pick.get("Cuota", 0) or 0)
    p_ml = float(pick.get("P_ML", 0) or 0)
    p_mc = float(pick.get("P_Estadistico", 0) or 0)
    p_combined = float(pick.get("P_Ensemble", 0) or 0)
    disagreement = float(pick.get("Desacuerdo_pp", abs(p_ml - p_mc)) or 0)
    ev = float(pick.get("EV_pct", 0) or 0)
    edge = float(pick.get("Edge_pp", 0) or 0)
    p_for_kelly = float(pick.get("P_Condicional", p_combined) or p_combined)
    kelly = _kelly_fraction(p_for_kelly, odds)
    stake = round(BANKROLL_MXN * kelly, 2)
    verdict = "🔥 Value Fuerte" if ev >= 8.0 and edge >= 5.0 else "✅ Value"
    record_key = "|".join([
        game_date, fixture_id, "Liga MX", away, home, selection, MODEL_VERSION
    ])
    return [
        record_key,
        now_utc,
        game_date,
        fixture_id,
        "Liga MX",
        away,
        home,
        market,
        selection,
        round(odds, 2),
        round(p_ml, 1),
        round(p_mc, 1),
        round(p_combined, 1),
        round(disagreement, 1),
        round(ev, 2),
        round(kelly * 100.0, 2),
        round(BANKROLL_MXN, 2),
        stake,
        verdict,
        MODEL_VERSION,
        "pending",
        "",
        "",
        "",
        "",
    ]


def guardar_picks_ligamx(picks):
    """Guarda picks nuevos en LigaMX_Picks sin duplicar record_key.

    Nunca debe romper el scanner: devuelve un diagnóstico aunque Sheets falle.
    """
    picks = list(picks or [])
    if not picks:
        return {"ok": True, "saved": 0, "skipped": 0, "message": "Sin picks para guardar"}
    try:
        existing = _existing_keys()
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rows = []
        skipped = 0
        for pick in picks:
            row = _pick_row(pick, now_utc)
            if row[0] in existing:
                skipped += 1
                continue
            existing.add(row[0])
            rows.append(row)
        if not rows:
            return {"ok": True, "saved": 0, "skipped": skipped, "message": "Todos los picks ya estaban registrados"}

        rng = quote(f"'{SHEET_NAME}'!A:Y", safe="")
        url = (
            f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{rng}:append"
            "?valueInputOption=USER_ENTERED&insertDataOption=INSERT_ROWS"
        )
        r = requests.post(
            url,
            headers=_headers(),
            json={"majorDimension": "ROWS", "values": rows},
            timeout=20,
        )
        r.raise_for_status()
        return {
            "ok": True,
            "saved": len(rows),
            "skipped": skipped,
            "message": f"{len(rows)} pick(s) guardados en {SHEET_NAME}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "saved": 0,
            "skipped": 0,
            "message": f"Google Sheets no pudo guardar: {type(exc).__name__}: {exc}",
        }
