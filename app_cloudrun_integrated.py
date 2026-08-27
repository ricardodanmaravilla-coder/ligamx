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

_lock = threading.Lock()
_state = {"df": None, "elo_table": None, "elo_map": None, "ml": None, "fixtures": None, "fixtures_at": 0.0}


def safe(v):
    if isinstance(v, dict): return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [safe(x) for x in v]
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return None if np.isnan(v) else float(v)
    if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)): return v.isoformat()
    try:
        if not isinstance(v, (str, bytes)) and pd.isna(v): return None
    except Exception:
        pass
    return v


def load_history():
    for path in ("data/historico_ligamx_completo.csv", "historico_ligamx_completo.csv"):
        try: return clean_history(pd.read_csv(path))
        except Exception: pass
    raise RuntimeError("No se pudo cargar el histórico de Liga MX")


def state():
    if _state["df"] is not None and _state["ml"] is not None: return _state
    with _lock:
        if _state["df"] is None:
            df = load_history()
            elo = SistemaEloLigaMX().calcular_historico(df)
            ml = PredictorML()
            if not ml.entrenar(df): raise RuntimeError("El modelo ML no pudo entrenar")
            _state.update({"df": df, "elo_table": elo, "elo_map": dict(zip(elo.Equipo, elo.ELO_Rating)), "ml": ml})
    return _state


def fixtures(force=False):
    now = time.time()
    if not force and _state["fixtures"] is not None and now-_state["fixtures_at"] < 300: return _state["fixtures"]
    if not API_KEY: raise RuntimeError("Falta API_SPORTS_KEY")
    today = datetime.date.today()
    params = {"league": LIGA_MX_ID, "season": LIGA_MX_SEASON, "from": today.isoformat(), "to": (today+datetime.timedelta(days=45)).isoformat(), "timezone": "America/Mexico_City"}
    r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params, timeout=15)
    r.raise_for_status(); payload = r.json()
    if payload.get("errors"): raise RuntimeError(f"API-Football: {payload['errors']}")
    out=[]
    for f in payload.get("response", []):
        status=f.get("fixture",{}).get("status",{}).get("short","")
        if status in {"FT","AET","PEN","CANC","ABD","PST"}: continue
        out.append({"fixture_id":f["fixture"]["id"],"fecha":f["fixture"]["date"][:16].replace("T"," "),"local":normalize_team(f["teams"]["home"]["name"]),"visita":normalize_team(f["teams"]["away"]["name"])})
    _state["fixtures"], _state["fixtures_at"] = out, now
    return out


def line(cuotas, kind, default):
    v=(cuotas or {}).get(f"linea_{kind}_detectada")
    return float(v) if v is not None else float(default)


def market_rows(mcsec, mlsec, over_key, under_key, push_key=None):
    rows=[]
    for side,key in (("Over",over_key),("Under",under_key)):
        if key in mcsec and key in mlsec:
            a=float(mcsec[key]); b=float(mlsec[key])
            rows.append({"Lado":side,"Monte Carlo %":round(a,1),"ML %":round(b,1),"Ensemble %":round(.55*a+.45*b,1),"Diferencia pp":round(abs(a-b),1)})
    if push_key and (push_key in mcsec or push_key in mlsec):
        rows.append({"Lado":"Push","Monte Carlo %":round(float(mcsec.get(push_key,0)),1),"ML %":round(float(mlsec.get(push_key,0)),1)})
    return rows


HTML=r'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Liga MX Analytics</title><style>
:root{--bg:#07111f;--p:#0d1b2a;--p2:#13263b;--t:#edf5ff;--m:#9fb3c8;--a:#38bdf8;--l:#27415c;--ok:#22c55e}*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Arial;background:#07111f;color:var(--t)}.w{max-width:1250px;margin:auto;padding:22px}.hero{display:flex;justify-content:space-between;gap:12px;align-items:center}.hero h1{font-size:clamp(28px,5vw,46px);margin:0}.muted{color:var(--m)}.badge{border:1px solid var(--l);padding:8px 12px;border-radius:999px}.g{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:18px}.c{grid-column:span 12;background:var(--p);border:1px solid var(--l);border-radius:16px;padding:17px}.third{grid-column:span 4}.half{grid-column:span 6}.quarter{grid-column:span 4}@media(max-width:850px){.third,.half,.quarter{grid-column:span 12}.hero,.row{flex-direction:column;align-items:stretch}}.row{display:flex;gap:10px}.row>*{flex:1}select,button{width:100%;padding:12px;border-radius:11px;border:1px solid var(--l);background:#0a1726;color:var(--t)}button{background:#0284c7;border:0;font-weight:700;cursor:pointer}.secondary{background:#172b40}.kpi{font-size:27px;font-weight:800}.status{padding:11px;border:1px solid var(--l);border-radius:11px;margin-top:10px}.tw{overflow:auto;max-height:470px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px;border-bottom:1px solid var(--l);text-align:left}th{background:var(--p);position:sticky;top:0;color:#b9d7ed}.tag{font-size:12px;color:#fbbf24}.ok{color:var(--ok)}
</style></head><body><div class="w"><div class="hero"><div><h1>Liga MX Analytics</h1><div class="muted">ELO · ML · Monte Carlo · 1X2 · goles · corners · tarjetas</div></div><div class="badge">Cloud Run · Mercados integrados</div></div><div class="g">
<section class="c"><h2>Partidos próximos</h2><div class="row"><select id="fixture"></select><button id="analyze">Analizar partido</button></div><div id="msg" class="status muted">Cargando...</div></section>
<section class="c third"><div class="muted">ELO local</div><div id="eh" class="kpi">—</div></section><section class="c third"><div class="muted">ELO visitante</div><div id="ea" class="kpi">—</div></section><section class="c third"><div class="muted">Diferencia</div><div id="ed" class="kpi">—</div></section>
<section class="c half"><h2>1X2</h2><div id="x12" class="tw"></div></section><section class="c half"><h2>Picks del modelo</h2><div class="tag">Incluye 1X2, goles, corners y tarjetas cuando superan los filtros internos.</div><div id="picks" class="tw"></div></section>
<section class="c quarter"><h2>Goles O/U</h2><div id="goals" class="tw"></div></section><section class="c quarter"><h2>Corners O/U</h2><div id="corners" class="tw"></div></section><section class="c quarter"><h2>Tarjetas O/U</h2><div id="cards" class="tw"></div></section>
<section class="c"><h2>Diagnóstico completo</h2><div id="diag" class="tw"></div></section>
<section class="c"><div class="row"><div><h2>Scanner de jornada</h2><div class="muted">Busca oportunidades en todos los mercados integrados.</div></div><button id="scan" class="secondary">Escanear jornada completa</button></div><div id="scanout" class="tw"></div></section>
<section class="c"><div class="row"><h2>Ranking ELO</h2><button id="elo" class="secondary">Actualizar ranking</button></div><div id="elot" class="tw"></div></section>
</div></div><script>
const $=s=>document.querySelector(s);function table(rows){if(!rows||!rows.length)return '<div class="status muted">Sin datos.</div>';let k=[...new Set(rows.flatMap(r=>Object.keys(r)))];return '<table><thead><tr>'+k.map(x=>`<th>${x}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+k.map(x=>`<td>${r[x]??''}</td>`).join('')+'</tr>').join('')+'</tbody></table>'}async function api(u,o){let r=await fetch(u,o),j=await r.json();if(!r.ok)throw Error(j.error||'Error');return j}
async function lf(){try{let j=await api('/api/fixtures');$('#fixture').innerHTML=j.fixtures.map(f=>`<option value="${f.fixture_id}">${f.fecha} | ${f.local} vs ${f.visita}</option>`).join('');$('#msg').textContent=`${j.fixtures.length} partido(s) encontrados`}catch(e){$('#msg').textContent=e.message}}
$('#analyze').onclick=async()=>{let b=$('#analyze');b.disabled=true;$('#msg').textContent='Calculando todos los mercados...';try{let j=await api('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fixture_id:Number($('#fixture').value)})});$('#msg').textContent=`${j.local} vs ${j.visita} — listo`;$('#eh').textContent=j.elo.local.toFixed(1);$('#ea').textContent=j.elo.visita.toFixed(1);$('#ed').textContent=(j.elo.diferencia>=0?'+':'')+j.elo.diferencia.toFixed(1);$('#x12').innerHTML=table(j.resultado_1x2);$('#goals').innerHTML=table(j.mercados.goles);$('#corners').innerHTML=table(j.mercados.corners);$('#cards').innerHTML=table(j.mercados.tarjetas);$('#picks').innerHTML=table(j.picks);$('#diag').innerHTML=table(j.diagnostics)}catch(e){$('#msg').textContent=e.message}finally{b.disabled=false}}
$('#scan').onclick=async()=>{let b=$('#scan');b.disabled=true;$('#scanout').innerHTML='<div class="status muted">Escaneando...</div>';try{let j=await api('/api/scan',{method:'POST'});$('#scanout').innerHTML=table(j.picks)}catch(e){$('#scanout').innerHTML=e.message}finally{b.disabled=false}}
async function le(){try{let j=await api('/api/elo');$('#elot').innerHTML=table(j.ranking)}catch(e){$('#elot').innerHTML=e.message}}$('#elo').onclick=le;lf();le();
</script></body></html>'''


@app.get("/")
def index(): return render_template_string(HTML)

@app.get("/health")
def health(): return jsonify({"status":"ok","service":"ligamx-cloudrun","markets":["1X2","goals","corners","cards"]})

@app.get("/api/fixtures")
def api_fixtures():
    try: return jsonify({"fixtures":safe(fixtures(request.args.get("refresh")=="1"))})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.post("/api/analyze")
def analyze():
    try:
        fid=int((request.get_json(silent=True) or {}).get("fixture_id")); f=next((x for x in fixtures() if int(x["fixture_id"])==fid),None)
        if not f: return jsonify({"error":"Fixture no encontrado"}),404
        s=state(); df,em,ml=s["df"],s["elo_map"],s["ml"]; local,away=f["local"],f["visita"]
        if local not in em or away not in em: return jsonify({"error":"Equipo sin ELO histórico suficiente"}),422
        q=obtener_cuotas_partido(fid); lg,lc,lt=line(q,"goles",2.5),line(q,"corners",9.5),line(q,"tarjetas",4.5)
        mc=simular_partido_montecarlo(local,away,df_historico=df,elo_local=em[local],elo_visita=em[away],linea_goles=lg,linea_corners=lc,linea_tarjetas=lt)
        mp=ml.predecir_mercados_completos(df,local,away,elo_local=em[local],elo_visita=em[away],linea_goles=lg,linea_corners=lc,linea_tarjetas=lt)
        x12=[]
        for m in ("Gana Local","Empate","Gana Visita"):
            a,b=float(mc["Resultado_1X2"][m]),float(mp["Resultado_1X2"][m]); x12.append({"Mercado":m,"Monte Carlo %":round(a,1),"ML %":round(b,1),"Ensemble %":round(.55*a+.45*b,1),"Diferencia pp":round(abs(a-b),1)})
        raw,diag=evaluar_fixture(local,away,fid,df,cuotas=q,ml=ml,elo_map=em,return_diagnostics=True)
        markets={
          "goles":market_rows(mc.get("Goles_Over_Under",{}),mp.get("Goles_Over_Under",{}),f"Over {lg}",f"Under {lg}",f"Push {lg}"),
          "corners":market_rows(mc.get("Corners_Totales",{}),mp.get("Corners_Totales",{}),f"Over {lc} Corners",f"Under {lc} Corners",f"Push {lc} Corners"),
          "tarjetas":market_rows(mc.get("Tarjetas_Totales",{}),mp.get("Tarjetas_Totales",{}),f"Over {lt} Tarjetas",f"Under {lt} Tarjetas",f"Push {lt} Tarjetas")}
        return jsonify(safe({"fixture_id":fid,"local":local,"visita":away,"elo":{"local":float(em[local]),"visita":float(em[away]),"diferencia":float(em[local]-em[away])},"lineas":{"goles":lg,"corners":lc,"tarjetas":lt},"resultado_1x2":x12,"mercados":markets,"picks":raw,"diagnostics":diag,"cuotas":q}))
    except Exception as e: return jsonify({"error":f"Error de análisis: {e}"}),500

@app.post("/api/scan")
def scan():
    try:
        raw=escanear_jornada_actual(state()["df"])
        return jsonify({"picks":safe(raw)})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.get("/api/elo")
def elo():
    try: return jsonify({"ranking":safe(state()["elo_table"].to_dict(orient="records"))})
    except Exception as e: return jsonify({"error":str(e)}),500

if __name__=="__main__": app.run(host="0.0.0.0",port=int(os.environ.get("PORT","8080")),threaded=True)
