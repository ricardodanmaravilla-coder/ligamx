import os
import requests

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LIGA_MX_ID = 262


def obtener_ultimos_fixtures_equipo(equipo_id, cantidad=10):
    """Devuelve partidos terminados recientes; sin Streamlit ni estado global oculto."""
    if not API_KEY or not equipo_id:
        return []
    try:
        res = requests.get(
            f"{BASE_URL}/fixtures",
            headers=HEADERS,
            params={"league": LIGA_MX_ID, "team": equipo_id, "last": cantidad, "status": "FT"},
            timeout=10,
        )
        if res.status_code != 200:
            return []
        return [p["fixture"]["id"] for p in res.json().get("response", [])]
    except Exception:
        return []


def _saves_for_fixture(fixture_id, equipo_id):
    try:
        res = requests.get(
            f"{BASE_URL}/fixtures/statistics",
            headers=HEADERS,
            params={"fixture": fixture_id, "team": equipo_id},
            timeout=10,
        )
        if res.status_code != 200:
            return None
        data = res.json().get("response", [])
        if not data:
            return None
        for stat in data[0].get("statistics", []):
            if stat.get("type") == "Goalkeeper Saves" and stat.get("value") is not None:
                return int(stat["value"])
    except Exception:
        pass
    return None


def _goals_allowed_for_fixture(fixture_id, equipo_id):
    try:
        res = requests.get(
            f"{BASE_URL}/fixtures",
            headers=HEADERS,
            params={"id": fixture_id},
            timeout=10,
        )
        if res.status_code != 200:
            return None
        data = res.json().get("response", [])
        if not data:
            return None
        p = data[0]
        if str(p["teams"]["home"]["id"]) == str(equipo_id):
            return int(p["goals"]["away"] or 0)
        return int(p["goals"]["home"] or 0)
    except Exception:
        return None


def calcular_eficiencia_portero_api(
    equipo_id,
    equipo_nombre=None,
    cantidad=10,
    promedio_liga=0.70,
    shrink_shots=30,
):
    """Factor conservador de rendimiento de porteria a nivel equipo.

    No pretende identificar al portero titular. Usa save% reciente del equipo y
    aplica shrinkage hacia 70% para reducir ruido por muestra pequena. El efecto
    queda limitado a +/-10%. Si faltan datos devuelve 1.0 (neutro).
    """
    fixtures = obtener_ultimos_fixtures_equipo(equipo_id, cantidad=cantidad)
    if len(fixtures) < 5:
        return 1.0

    saves = 0
    goals = 0
    valid = 0
    for fid in fixtures:
        s = _saves_for_fixture(fid, equipo_id)
        g = _goals_allowed_for_fixture(fid, equipo_id)
        if s is None or g is None:
            continue
        saves += s
        goals += g
        valid += 1

    shots_on_target_against = saves + goals
    if valid < 5 or shots_on_target_against <= 0:
        return 1.0

    # Beta/binomial style shrinkage expresado como tiros equivalentes de liga.
    shrunk_save_pct = (
        saves + shrink_shots * promedio_liga
    ) / (shots_on_target_against + shrink_shots)

    # Factor sobre goles esperados rivales. Mucho menos agresivo que V1.
    raw = (1.0 - shrunk_save_pct) / (1.0 - promedio_liga)
    return round(max(0.90, min(1.10, raw)), 3)


def obtener_stats_portero_partido(fixture_id, equipo_id):
    """Compatibilidad: devuelve atajadas observadas o 0 si faltan datos."""
    value = _saves_for_fixture(fixture_id, equipo_id)
    return int(value) if value is not None else 0
