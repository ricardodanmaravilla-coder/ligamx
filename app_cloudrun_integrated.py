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
from modules.scanner_engine import evaluar_fixture
from modules.sheets_ligamx import guardar_picks_ligamx

app = Flask(__name__)
API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}
LIGA_MX_ID = 262
LIGA_MX_SEASON = 2026
SCANNER_MAX_FIXTURES = 9

_lock = threading.Lock()
_state = {
    "df": None,
    "elo_table": None,
    "elo_map": None,
    "ml": None,
    "fixtures": None,
    "fixtures_at": 0.0,
}


def safe(v):
    if isinstance(v, dict):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
        return v.isoformat()
    try:
        if not isinstance(v, (str, bytes)) and pd.isna(v):
            return None
    except Exception:
        pass
    return v


def load_history():
    for path in ("data/historico_ligamx_completo.csv", "historico_ligamx_completo.csv"):
        try:
            return clean_history(pd.read_csv(path))
        except Exception:
            pass
    raise RuntimeError("No se pudo cargar el histórico de Liga MX")


def model_state():
    if _state["df"] is not None and _state["ml"] is not None:
        return _state
    with _lock:
        if _state["df"] is None:
            df = load_history()
            elo = SistemaEloLigaMX().calcular_historico(df)
            ml = PredictorML()
            if not ml.entrenar(df):
                raise RuntimeError("El modelo ML no pudo entrenar")
            _state.update({
                "df": df,
                "elo_table": elo,
                "elo_map": dict(zip(elo.Equipo, elo.ELO_Rating)),
                "ml": ml,
            })
    return _state


def fixtures(force=False):
    now = time.time()
    if (
        not force
        and _state["fixtures"] is not None
        and now - _state["fixtures_at"] < 300
    ):
        return _state["fixtures"]
    if not API_KEY:
        raise RuntimeError("Falta API_SPORTS_KEY")

    today = datetime.date.today()
    params = {
        "league": LIGA_MX_ID,
        "season": LIGA_MX_SEASON,
        "from": today.isoformat(),
        "to": (today + datetime.timedelta(days=45)).isoformat(),
        "timezone": "America/Mexico_City",
    }
    r = requests.get(
        f"{BASE_URL}/fixtures", headers=HEADERS, params=params, timeout=15
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(f"API-Football: {payload['errors']}")

    out = []
    for f in payload.get("response", []):
        status = f.get("fixture", {}).get("status", {}).get("short", "")
        if status in {"FT", "AET", "PEN", "CANC", "ABD", "PST"}:
            continue
        out.append({
            "fixture_id": f["fixture"]["id"],
            "fecha": f["fixture"]["date"][:16].replace("T", " "),
            "local": normalize_team(f["teams"]["home"]["name"]),
            "visita": normalize_team(f["teams"]["away"]["name"]),
        })
    out.sort(key=lambda x: (x.get("fecha", ""), int(x.get("fixture_id", 0))))
    _state["fixtures"] = out
    _state["fixtures_at"] = now
    return out


def line(cuotas, kind, default):
    v = (cuotas or {}).get(f"linea_{kind}_detectada")
    return float(v) if v is not None else float(default)


def market_rows(mcsec, mlsec, over_key, under_key, push_key=None):
    rows = []
    for side, key in (("Over", over_key), ("Under", under_key)):
        if key in mcsec and key in mlsec:
            a = float(mcsec[key])
            b = float(mlsec[key])
            rows.append({
                "Lado": side,
                "Monte Carlo %": round(a, 1),
                "ML %": round(b, 1),
                "Ensemble %": round(0.55 * a + 0.45 * b, 1),
                "Diferencia pp": round(abs(a - b), 1),
            })
    if push_key and (push_key in mcsec or push_key in mlsec):
        rows.append({
            "Lado": "Push",
            "Monte Carlo %": round(float(mcsec.get(push_key, 0)), 1),
            "ML %": round(float(mlsec.get(push_key, 0)), 1),
        })
    return rows


def enrich_pick(pick, fixture):
    p = dict(pick)
    p["fixture_id"] = int(fixture["fixture_id"])
    p["game_date"] = str(fixture["fecha"])[:10]
    p["home"] = fixture["local"]
    p["away"] = fixture["visita"]
    p["Local"] = fixture["local"]
    p["Visita"] = fixture["visita"]
    return p


HTML = r'''<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Liga MX Analytics</title>
<style>
:root{--bg:#07111f;--p:#0d1b2a;--p2:#13263b;--t:#edf5ff;--m:#9fb3c8;--a:#38bdf8;--l:#27415c;--ok:#22c55e;--hot:#f59e0b;--bad:#ef4444}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Arial;background:#07111f;color:var(--t)}
.w{max-width:1250px;margin:auto;padding:18px}.hero{display:flex;justify-content:space-between;gap:12px;align-items:center}.hero h1{font-size:clamp(28px,5vw,46px);margin:0}.muted{color:var(--m)}.badge{border:1px solid var(--l);padding:8px 12px;border-radius:999px}.g{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:18px}.c{grid-column:span 12;background:var(--p);border:1px solid var(--l);border-radius:16px;padding:17px}.third{grid-column:span 4}.half{grid-column:span 6}.quarter{grid-column:span 4}.row{display:flex;gap:10px;align-items:center}.row>*{flex:1}select,button{width:100%;padding:12px;border-radius:11px;border:1px solid var(--l);background:#0a1726;color:var(--t);font-size:15px}button{background:#0284c7;border:0;font-weight:750;cursor:pointer}.secondary{background:#172b40}.kpi{font-size:27px;font-weight:800}.status{padding:11px;border:1px solid var(--l);border-radius:11px;margin-top:10px}.tw{overflow:auto;max-height:470px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px;border-bottom:1px solid var(--l);text-align:left}th{background:var(--p);position:sticky;top:0;color:#b9d7ed}.tag{font-size:12px;color:#fbbf24}.ok{color:var(--ok)}
.pickgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:12px}.pick{background:linear-gradient(145deg,#10253a,#0a1928);border:1px solid #2a4967;border-radius:16px;padding:16px}.pick.hot{border-color:#7c5b19;box-shadow:0 0 0 1px rgba(245,158,11,.1)}.picktop{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}.pickgame{font-size:14px;color:#b9cad9;margin-bottom:8px}.pickmarket{font-size:22px;font-weight:900;line-height:1.15}.pickodd{font-size:22px;font-weight:900;color:#7dd3fc;white-space:nowrap}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:14px}.stat{background:#081523;border-radius:10px;padding:9px}.stat b{display:block;font-size:16px}.stat span{font-size:11px;color:var(--m)}.book{margin-top:10px;font-size:12px;color:var(--m)}.empty{padding:18px;text-align:center;border:1px dashed var(--l);border-radius:12px;color:var(--m)}
@media(max-width:850px){.third,.half,.quarter{grid-column:span 12}.hero,.row{flex-direction:column;align-items:stretch}.pickgrid{grid-template-columns:1fr}.w{padding:12px}.c{padding:15px}.pickmarket{font-size:21px}.stats{grid-template-columns:repeat(3,1fr)}}
</style>
</head>
<body><div class="w">
<div class="hero"><div><h1>Liga MX Analytics</h1><div class="muted">ELO · ML · Monte Carlo · cuotas reales · picks claros</div></div><div class="badge">Cloud Run · V4</div></div>
<div class="g">
<section class="c"><h2>Partidos próximos</h2><div class="row"><select id="fixture"></select><button id="analyze">Analizar partido</button></div><div id="msg" class="status muted">Cargando...</div></section>
<section class="c third"><div class="muted">ELO local</div><div id="eh" class="kpi">—</div></section><section class="c third"><div class="muted">ELO visitante</div><div id="ea" class="kpi">—</div></section><section class="c third"><div class="muted">Diferencia</div><div id="ed" class="kpi">—</div></section>
<section class="c"><h2>🔥 Picks recomendados</h2><div class="muted">Formato directo: selección, cuota, probabilidad, edge y EV.</div><div id="picks"></div></section>
<section class="c half"><h2>1X2</h2><div id="x12" class="tw"></div></section>
<section class="c quarter"><h2>Goles O/U</h2><div id="goals" class="tw"></div></section><section class="c quarter"><h2>Corners O/U</h2><div id="corners" class="tw"></div></section><section class="c quarter"><h2>Tarjetas O/U</h2><div id="cards" class="tw"></div></section>
<section class="c"><h2>Diagnóstico completo</h2><div id="diag" class="tw"></div></section>
<section class="c"><h2>🌐 Scanner de jornada</h2><div class="muted">Analiza exclusivamente los 9 partidos más próximos de Liga MX.</div><button id="scan" class="secondary" style="margin-top:12px">Escanear jornada de 9 partidos</button><div id="scanstatus" class="status muted">Listo para escanear.</div><div id="scanout"></div></section>
<section class="c"><div class="row"><h2>Ranking ELO</h2><button id="elo" class="secondary">Actualizar ranking</button></div><div id="elot" class="tw"></div></section>
</div></div>
<script>
const $=s=>document.querySelector(s);
function table(rows){if(!rows||!rows.length)return '<div class="empty">Sin datos.</div>';let k=[...new Set(rows.flatMap(r=>Object.keys(r)))];return '<table><thead><tr>'+k.map(x=>`<th>${x}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+k.map(x=>`<td>${r[x]??''}</td>`).join('')+'</tr>').join('')+'</tbody></table>'}
function esc(s){return String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
function pickCards(rows){if(!rows||!rows.length)return '<div class="empty">NO BET — ningún mercado superó los filtros.</div>';return '<div class="pickgrid">'+rows.map(p=>{let ev=Number(p.EV_pct||0),edge=Number(p.Edge_pp||0),prob=Number(p.P_Condicional||p.P_Ensemble||0),odd=Number(p.Cuota||0),hot=ev>=8&&edge>=5;return `<article class="pick ${hot?'hot':''}"><div class="pickgame">${esc(p.Partido||((p.home||'')+' vs '+(p.away||'')))}</div><div class="picktop"><div class="pickmarket">${hot?'🔥':'✅'} ${esc(p.Mercado||'PICK')}</div><div class="pickodd">@ ${odd?odd.toFixed(2):'—'}</div></div><div class="stats"><div class="stat"><b>${prob.toFixed(1)}%</b><span>Probabilidad</span></div><div class="stat"><b>+${edge.toFixed(1)}</b><span>Edge pp</span></div><div class="stat"><b>+${ev.toFixed(1)}%</b><span>EV</span></div></div><div class="book">Bookmaker: ${esc(p.Bookmaker||'N/D')} · MC ${Number(p.P_Estadistico||0).toFixed(1)}% · ML ${Number(p.P_ML||0).toFixed(1)}%</div></article>`}).join('')+'</div>'}
async function api(u,o){let r=await fetch(u,o),j=await r.json();if(!r.ok)throw Error(j.error||'Error');return j}
async function lf(){try{let j=await api('/api/fixtures');$('#fixture').innerHTML=j.fixtures.map(f=>`<option value="${f.fixture_id}">${f.fecha} | ${f.local} vs ${f.visita}</option>`).join('');$('#msg').textContent=`${j.fixtures.length} partido(s) futuros disponibles. El scanner usa sólo los próximos 9.`}catch(e){$('#msg').textContent=e.message}}
$('#analyze').onclick=async()=>{let b=$('#analyze');b.disabled=true;$('#msg').textContent='Calculando todos los mercados...';try{let j=await api('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fixture_id:Number($('#fixture').value)})});$('#msg').textContent=`${j.local} vs ${j.visita} — análisis listo`;$('#eh').textContent=j.elo.local.toFixed(1);$('#ea').textContent=j.elo.visita.toFixed(1);$('#ed').textContent=(j.elo.diferencia>=0?'+':'')+j.elo.diferencia.toFixed(1);$('#x12').innerHTML=table(j.resultado_1x2);$('#goals').innerHTML=table(j.mercados.goles);$('#corners').innerHTML=table(j.mercados.corners);$('#cards').innerHTML=table(j.mercados.tarjetas);$('#picks').innerHTML=pickCards(j.picks);$('#diag').innerHTML=table(j.diagnostics)}catch(e){$('#msg').textContent=e.message}finally{b.disabled=false}}
$('#scan').onclick=async()=>{let b=$('#scan');b.disabled=true;$('#scanstatus').textContent='Analizando los próximos 9 partidos...';$('#scanout').innerHTML='';try{let j=await api('/api/scan',{method:'POST'});$('#scanstatus').textContent=`${j.partidos_analizados} partido(s) analizados · ${j.picks.length} pick(s). ${j.sheet?.message||''}`;$('#scanout').innerHTML=pickCards(j.picks)}catch(e){$('#scanstatus').textContent=e.message}finally{b.disabled=false}}
async function le(){try{let j=await api('/api/elo');$('#elot').innerHTML=table(j.ranking)}catch(e){$('#elot').innerHTML=e.message}}$('#elo').onclick=le;lf();le();
</script></body></html>'''


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "ligamx-cloudrun",
        "version": "v4",
        "scanner_max_fixtures": SCANNER_MAX_FIXTURES,
        "markets": ["1X2", "goals", "corners", "cards"],
    })


@app.get("/api/fixtures")
def api_fixtures():
    try:
        return jsonify({"fixtures": safe(fixtures(request.args.get("refresh") == "1"))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/analyze")
def analyze():
    try:
        fid = int((request.get_json(silent=True) or {}).get("fixture_id"))
        f = next((x for x in fixtures() if int(x["fixture_id"]) == fid), None)
        if not f:
            return jsonify({"error": "Fixture no encontrado"}), 404
        s = model_state()
        df, em, ml = s["df"], s["elo_map"], s["ml"]
        local, away = f["local"], f["visita"]
        if local not in em or away not in em:
            return jsonify({"error": "Equipo sin ELO histórico suficiente"}), 422

        q = obtener_cuotas_partido(fid)
        lg, lc, lt = line(q, "goles", 2.5), line(q, "corners", 9.5), line(q, "tarjetas", 4.5)
        mc = simular_partido_montecarlo(
            local, away, df_historico=df,
            elo_local=em[local], elo_visita=em[away],
            linea_goles=lg, linea_corners=lc, linea_tarjetas=lt,
        )
        mp = ml.predecir_mercados_completos(
            df, local, away,
            elo_local=em[local], elo_visita=em[away],
            linea_goles=lg, linea_corners=lc, linea_tarjetas=lt,
        )

        x12 = []
        for m in ("Gana Local", "Empate", "Gana Visita"):
            a = float(mc["Resultado_1X2"][m])
            b = float(mp["Resultado_1X2"][m])
            x12.append({
                "Mercado": m,
                "Monte Carlo %": round(a, 1),
                "ML %": round(b, 1),
                "Ensemble %": round(0.55 * a + 0.45 * b, 1),
                "Diferencia pp": round(abs(a - b), 1),
            })

        raw, diag = evaluar_fixture(
            local, away, fid, df,
            cuotas=q, ml=ml, elo_map=em, return_diagnostics=True,
        )
        raw = [enrich_pick(p, f) for p in raw]
        markets = {
            "goles": market_rows(
                mc.get("Goles_Over_Under", {}), mp.get("Goles_Over_Under", {}),
                f"Over {lg}", f"Under {lg}", f"Push {lg}",
            ),
            "corners": market_rows(
                mc.get("Corners_Totales", {}), mp.get("Corners_Totales", {}),
                f"Over {lc} Corners", f"Under {lc} Corners", f"Push {lc} Corners",
            ),
            "tarjetas": market_rows(
                mc.get("Tarjetas_Totales", {}), mp.get("Tarjetas_Totales", {}),
                f"Over {lt} Tarjetas", f"Under {lt} Tarjetas", f"Push {lt} Tarjetas",
            ),
        }
        return jsonify(safe({
            "fixture_id": fid,
            "local": local,
            "visita": away,
            "elo": {
                "local": float(em[local]),
                "visita": float(em[away]),
                "diferencia": float(em[local] - em[away]),
            },
            "lineas": {"goles": lg, "corners": lc, "tarjetas": lt},
            "resultado_1x2": x12,
            "mercados": markets,
            "picks": raw,
            "diagnostics": diag,
            "cuotas": q,
        }))
    except Exception as e:
        return jsonify({"error": f"Error de análisis: {e}"}), 500


@app.post("/api/scan")
def scan():
    try:
        s = model_state()
        df, em, ml = s["df"], s["elo_map"], s["ml"]
        upcoming = fixtures(force=True)[:SCANNER_MAX_FIXTURES]
        all_picks = []
        errors = []
        for f in upcoming:
            try:
                local, away, fid = f["local"], f["visita"], int(f["fixture_id"])
                if local not in em or away not in em:
                    errors.append(f"{local} vs {away}: sin ELO suficiente")
                    continue
                picks = evaluar_fixture(
                    local, away, fid, df,
                    ml=ml, elo_map=em,
                )
                all_picks.extend(enrich_pick(p, f) for p in picks)
            except Exception as exc:
                errors.append(f"{f.get('local','?')} vs {f.get('visita','?')}: {type(exc).__name__}: {exc}")

        all_picks.sort(
            key=lambda x: (float(x.get("EV_pct", 0) or 0), float(x.get("Edge_pp", 0) or 0)),
            reverse=True,
        )
        sheet_status = guardar_picks_ligamx(all_picks)
        return jsonify(safe({
            "partidos_analizados": len(upcoming),
            "limite_partidos": SCANNER_MAX_FIXTURES,
            "picks": all_picks,
            "sheet": sheet_status,
            "errores": errors[:20],
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/elo")
def api_elo():
    try:
        ranking = model_state()["elo_table"].to_dict(orient="records")
        return jsonify({"ranking": safe(ranking)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), threaded=True)
