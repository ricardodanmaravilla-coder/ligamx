import datetime
import os
import threading
import time

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request

from modules.feature_engineering import clean_history, normalize_team
from modules.elo_engine import SistemaEloLigaMX
from modules.ml_engine import PredictorML
from modules.montecarlo_sim import simular_partido_montecarlo
from modules.odds_engine import obtener_cuotas_partido
from modules.scanner_engine import evaluar_fixture, escanear_jornada_actual

app = Flask(__name__)

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}
LIGA_MX_ID = 262
LIGA_MX_SEASON = 2026

_state_lock = threading.Lock()
_state = {
    "df": None,
    "elo_table": None,
    "elo_map": None,
    "ml": None,
    "loaded_at": 0.0,
    "fixtures": None,
    "fixtures_at": 0.0,
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def _load_history():
    for path in ["data/historico_ligamx_completo.csv", "historico_ligamx_completo.csv"]:
        try:
            return clean_history(pd.read_csv(path))
        except Exception:
            continue
    raise RuntimeError("No se pudo cargar el histórico de Liga MX")


def get_model_state():
    if _state["df"] is not None and _state["ml"] is not None:
        return _state

    with _state_lock:
        if _state["df"] is not None and _state["ml"] is not None:
            return _state

        df = _load_history()
        elo_table = SistemaEloLigaMX().calcular_historico(df)
        elo_map = dict(zip(elo_table.Equipo, elo_table.ELO_Rating))
        ml = PredictorML()
        if not ml.entrenar(df):
            raise RuntimeError("El modelo ML no pudo entrenar")

        _state.update({
            "df": df,
            "elo_table": elo_table,
            "elo_map": elo_map,
            "ml": ml,
            "loaded_at": time.time(),
        })
    return _state


def get_fixtures(force=False):
    now = time.time()
    if not force and _state["fixtures"] is not None and now - _state["fixtures_at"] < 300:
        return _state["fixtures"]

    if not API_KEY:
        raise RuntimeError("Falta la variable API_SPORTS_KEY")

    today = datetime.date.today()
    params = {
        "league": LIGA_MX_ID,
        "season": LIGA_MX_SEASON,
        "from": today.strftime("%Y-%m-%d"),
        "to": (today + datetime.timedelta(days=45)).strftime("%Y-%m-%d"),
        "timezone": "America/Mexico_City",
    }
    r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(f"API-Football: {payload.get('errors')}")

    out = []
    for fixture in payload.get("response", []):
        status = fixture.get("fixture", {}).get("status", {}).get("short", "")
        if status in {"FT", "AET", "PEN", "CANC", "ABD", "PST"}:
            continue
        out.append({
            "fixture_id": fixture["fixture"]["id"],
            "fecha": fixture["fixture"]["date"][:16].replace("T", " "),
            "local": normalize_team(fixture["teams"]["home"]["name"]),
            "visita": normalize_team(fixture["teams"]["away"]["name"]),
        })

    _state["fixtures"] = out
    _state["fixtures_at"] = now
    return out


def _line_or_default(cuotas, kind, default):
    value = cuotas.get(f"linea_{kind}_detectada") if cuotas else None
    return float(value) if value is not None else default


def _filtrar_picks_publicables(picks):
    publicados = []
    for p in picks or []:
        mercado = str(p.get("Mercado", ""))
        if "Corners O/U" in mercado or "Tarjetas O/U" in mercado:
            continue
        if "Goles O/U" in mercado:
            if float(p.get("P_Condicional", 0) or 0) < 65.0:
                continue
            if float(p.get("Edge_pp", 0) or 0) < 5.0 or float(p.get("EV_pct", 0) or 0) < 4.0:
                continue
        publicados.append(p)
    return publicados


def _ajustar_diagnosticos(diagnostics):
    out = []
    for item in diagnostics or []:
        d = dict(item)
        mercado = str(d.get("Mercado", ""))
        if "Corners O/U" in mercado or "Tarjetas O/U" in mercado:
            d["Estado"] = "ANÁLISIS SOLO"
            d["Motivo"] = "Mercado integrado, pero no habilitado como pick: backtest selectivo aún insuficiente"
        elif "Goles O/U" in mercado and str(d.get("Estado", "")).startswith("VALUE"):
            if float(d.get("P_Condicional", 0) or 0) < 65.0:
                d["Estado"] = "NO BET"
                d["Motivo"] = "Filtro final V3: goles O/U exige probabilidad condicional >=65%"
        out.append(d)
    return out


INDEX_HTML = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Liga MX Analytics V3</title>
  <style>
    :root{--bg:#07111f;--panel:#0d1b2a;--panel2:#13263b;--text:#edf5ff;--muted:#9fb3c8;--ok:#22c55e;--warn:#f59e0b;--accent:#38bdf8;--line:#27415c}
    *{box-sizing:border-box} body{margin:0;font-family:Inter,system-ui,Arial;background:linear-gradient(180deg,#06101d,#091827 45%,#07111f);color:var(--text)}
    .wrap{max-width:1200px;margin:auto;padding:24px}.hero{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:20px}.hero h1{margin:0;font-size:clamp(28px,5vw,48px)}
    .sub{color:var(--muted);margin-top:8px}.badge{padding:8px 12px;border:1px solid var(--line);border-radius:999px;background:#0a1726;color:#bde7ff}
    .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}.card{grid-column:span 12;background:rgba(13,27,42,.92);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 12px 35px rgba(0,0,0,.2)}
    .third{grid-column:span 4}.half{grid-column:span 6}@media(max-width:800px){.third,.half{grid-column:span 12}.hero{align-items:flex-start;flex-direction:column}}
    select,button{width:100%;padding:13px 14px;border-radius:12px;border:1px solid var(--line);background:#0a1726;color:var(--text);font-size:15px}button{cursor:pointer;font-weight:700;background:linear-gradient(135deg,#0284c7,#0ea5e9);border:0}button.secondary{background:#172b40;border:1px solid var(--line)}
    button:disabled{opacity:.55;cursor:wait}.row{display:flex;gap:12px}.row>*{flex:1}.muted{color:var(--muted)}.status{margin-top:12px;padding:12px;border-radius:12px;background:#0a1726;border:1px solid var(--line)}
    table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left}th{color:#b9d7ed;position:sticky;top:0;background:var(--panel)}.tablewrap{overflow:auto;max-height:480px}
    .kpi{font-size:28px;font-weight:800}.ok{color:var(--ok)}.warn{color:var(--warn)}pre{white-space:pre-wrap;word-break:break-word;color:#c7d9eb}.hidden{display:none}
  </style>
</head>
<body>
<div class="wrap">
  <div class="hero"><div><h1>Liga MX Analytics</h1><div class="sub">ELO · Machine Learning · Monte Carlo · cuotas reales · value betting</div></div><div class="badge">Cloud Run Edition</div></div>
  <div class="grid">
    <section class="card"><h2>Partidos próximos</h2><div class="row"><select id="fixture"></select><button id="analyze">Analizar partido V3</button></div><div id="msg" class="status muted">Cargando partidos...</div></section>
    <section class="card third"><div class="muted">ELO local</div><div id="eloHome" class="kpi">—</div></section>
    <section class="card third"><div class="muted">ELO visitante</div><div id="eloAway" class="kpi">—</div></section>
    <section class="card third"><div class="muted">Diferencia ELO</div><div id="eloDiff" class="kpi">—</div></section>
    <section class="card half"><h2>1X2</h2><div id="oneXtwo" class="tablewrap"></div></section>
    <section class="card half"><h2>Picks publicables</h2><div id="picks" class="tablewrap"><div class="muted">Analiza un partido para ver resultados.</div></div></section>
    <section class="card"><h2>Diagnóstico completo</h2><div id="diag" class="tablewrap"></div></section>
    <section class="card"><div class="row"><div><h2>Scanner de jornada</h2><div class="muted">Escanea próximos partidos sin recargar toda la aplicación.</div></div><div><button id="scan" class="secondary">Escanear jornada completa</button></div></div><div id="scanOut" class="tablewrap" style="margin-top:14px"></div></section>
    <section class="card"><div class="row"><div><h2>Ranking ELO</h2></div><div><button id="eloBtn" class="secondary">Actualizar ranking</button></div></div><div id="eloTable" class="tablewrap" style="margin-top:14px"></div></section>
  </div>
</div>
<script>
const $=s=>document.querySelector(s);
function table(rows){if(!rows||!rows.length)return '<div class="muted">Sin datos.</div>';const keys=[...new Set(rows.flatMap(r=>Object.keys(r)))];return '<table><thead><tr>'+keys.map(k=>`<th>${k}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+keys.map(k=>`<td>${r[k]??''}</td>`).join('')+'</tr>').join('')+'</tbody></table>'}
async function api(url,opt){const r=await fetch(url,opt);const j=await r.json();if(!r.ok)throw new Error(j.error||'Error del servidor');return j}
async function loadFixtures(){try{const j=await api('/api/fixtures');$('#fixture').innerHTML=j.fixtures.map((f,i)=>`<option value="${f.fixture_id}" data-i="${i}">${f.fecha} | ${f.local} vs ${f.visita}</option>`).join('');window.fixtures=j.fixtures;$('#msg').textContent=j.fixtures.length?`${j.fixtures.length} partido(s) encontrados.`:'No hay partidos próximos.'}catch(e){$('#msg').textContent=e.message}}
$('#analyze').onclick=async()=>{const b=$('#analyze');b.disabled=true;$('#msg').textContent='Calculando ELO, ML, Monte Carlo y cuotas...';try{const id=Number($('#fixture').value);const j=await api('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fixture_id:id})});$('#msg').textContent=`${j.local} vs ${j.visita} — análisis listo`;$('#eloHome').textContent=j.elo.local.toFixed(1);$('#eloAway').textContent=j.elo.visita.toFixed(1);$('#eloDiff').textContent=(j.elo.diferencia>=0?'+':'')+j.elo.diferencia.toFixed(1);$('#oneXtwo').innerHTML=table(j.resultado_1x2);$('#picks').innerHTML=j.picks.length?table(j.picks):'<div class="status">NO BET — ninguna opción superó todos los filtros.</div>';$('#diag').innerHTML=table(j.diagnostics)}catch(e){$('#msg').textContent=e.message}finally{b.disabled=false}}
$('#scan').onclick=async()=>{const b=$('#scan');b.disabled=true;$('#scanOut').innerHTML='<div class="muted">Escaneando jornada...</div>';try{const j=await api('/api/scan',{method:'POST'});$('#scanOut').innerHTML=j.picks.length?table(j.picks):'<div class="status">NO BET — no hay oportunidades publicables.</div>'}catch(e){$('#scanOut').innerHTML=`<div class="status">${e.message}</div>`}finally{b.disabled=false}}
$('#eloBtn').onclick=loadElo;async function loadElo(){try{const j=await api('/api/elo');$('#eloTable').innerHTML=table(j.ranking)}catch(e){$('#eloTable').innerHTML=e.message}}
loadFixtures();loadElo();
</script>
</body></html>
"""


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ligamx-cloudrun"})


@app.get("/api/fixtures")
def api_fixtures():
    try:
        fixtures = get_fixtures(force=request.args.get("refresh") == "1")
        return jsonify({"fixtures": _json_safe(fixtures)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/analyze")
def api_analyze():
    try:
        body = request.get_json(silent=True) or {}
        fixture_id = int(body.get("fixture_id"))
        fixtures = get_fixtures()
        partido = next((f for f in fixtures if int(f["fixture_id"]) == fixture_id), None)
        if not partido:
            return jsonify({"error": "Fixture no encontrado"}), 404

        state = get_model_state()
        df, elo_map, ml = state["df"], state["elo_map"], state["ml"]
        local, visita = partido["local"], partido["visita"]
        if local not in elo_map or visita not in elo_map:
            return jsonify({"error": "Equipo sin ELO histórico suficiente: NO BET"}), 422

        cuotas = obtener_cuotas_partido(fixture_id)
        lg = _line_or_default(cuotas, "goles", 2.5)
        lc = _line_or_default(cuotas, "corners", 9.5)
        lt = _line_or_default(cuotas, "tarjetas", 4.5)

        mc = simular_partido_montecarlo(
            local, visita, df_historico=df,
            elo_local=elo_map[local], elo_visita=elo_map[visita],
            linea_goles=lg, linea_corners=lc, linea_tarjetas=lt,
        )
        mlp = ml.predecir_mercados_completos(
            df, local, visita,
            elo_local=elo_map[local], elo_visita=elo_map[visita],
            linea_goles=lg, linea_corners=lc, linea_tarjetas=lt,
        )

        rows = []
        for market in ["Gana Local", "Empate", "Gana Visita"]:
            pmc = float(mc["Resultado_1X2"][market])
            pml = float(mlp["Resultado_1X2"][market])
            rows.append({
                "Mercado": market,
                "Monte Carlo %": round(pmc, 1),
                "ML %": round(pml, 1),
                "Diferencia pp": round(abs(pmc - pml), 1),
            })

        raw_picks, diagnostics = evaluar_fixture(
            local, visita, fixture_id, df,
            cuotas=cuotas, ml=ml, elo_map=elo_map,
            return_diagnostics=True,
        )
        picks = _filtrar_picks_publicables(raw_picks)
        diagnostics = _ajustar_diagnosticos(diagnostics)

        return jsonify(_json_safe({
            "fixture_id": fixture_id,
            "local": local,
            "visita": visita,
            "elo": {
                "local": float(elo_map[local]),
                "visita": float(elo_map[visita]),
                "diferencia": float(elo_map[local] - elo_map[visita]),
            },
            "resultado_1x2": rows,
            "picks": picks,
            "diagnostics": diagnostics,
            "cuotas": cuotas,
            "totales_ml": mlp.get("Prediccion_Totales", {}),
        }))
    except Exception as exc:
        return jsonify({"error": f"NO BET / error de validación: {exc}"}), 500


@app.post("/api/scan")
def api_scan():
    try:
        state = get_model_state()
        raw = escanear_jornada_actual(state["df"])
        picks = _filtrar_picks_publicables(raw)
        return jsonify({"picks": _json_safe(picks)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/elo")
def api_elo():
    try:
        state = get_model_state()
        ranking = state["elo_table"].to_dict(orient="records")
        return jsonify({"ranking": _json_safe(ranking)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)
