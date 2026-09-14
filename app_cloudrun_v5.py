from flask import jsonify

import app_cloudrun_integrated as core
import modules.scanner_engine as scanner_module
from modules.elo_engine import SistemaEloLigaMX
from modules.ml_engine import PredictorML
from modules.montecarlo_sim import simular_partido_montecarlo as _mc_original
from modules.parquet_cache import (
    load_history_fast, load_prepared_features, load_mc_context, cache_info,
)
from modules.settler_ligamx import liquidar_picks_pendientes

app = core.app


PRODUCTION_MODEL_VERSION = "ligamx-v5-conservative"


def _is_quarter_line(text):
    """Bloquea temporalmente .25/.75 hasta tener settlement asiático end-to-end."""
    s = str(text or "")
    return ".25" in s or ".75" in s


def production_filter(picks):
    """Política de publicación derivada de la auditoría de calibración.

    - 1X2 conserva sus filtros internos.
    - Goles O/U exige probabilidad efectiva >=65%, edge >=5 pp y EV >=4%.
    - Corners y tarjetas quedan sólo para análisis hasta validar calibración.
    - Líneas .25/.75 no se publican mientras el cobrador no liquide medias ganancias/pérdidas.
    - Edge >20 pp se considera posible descalibración y no se publica.
    """
    out = []
    for p in picks or []:
        market = str(p.get("Mercado", ""))
        if "Corners O/U" in market or "Tarjetas O/U" in market:
            continue
        if _is_quarter_line(market):
            continue

        edge = float(p.get("Edge_pp", 0) or 0)
        ev = float(p.get("EV_pct", 0) or 0)
        prob = float(p.get("P_Condicional", p.get("P_Ensemble", 0)) or 0)

        if edge > 20.0:
            continue
        if "Goles O/U" in market:
            if prob < 65.0 or edge < 5.0 or ev < 4.0:
                continue

        row = dict(p)
        row["model_version"] = PRODUCTION_MODEL_VERSION
        out.append(row)
    return out


def optimized_model_state():
    """Carga histórico/features/contexto MC desde Parquet y precalcula forma una sola vez."""
    if core._state["df"] is not None and core._state["ml"] is not None:
        return core._state
    with core._lock:
        if core._state["df"] is None:
            df = load_history_fast()
            elo = SistemaEloLigaMX().calcular_historico(df)
            elo_map = dict(zip(elo.Equipo, elo.ELO_Rating))
            ml = PredictorML()
            prepared = load_prepared_features()
            if prepared is not None:
                trained = ml.entrenar_preparado(prepared)
                if trained:
                    ml.set_current_context(df)
            else:
                trained = ml.entrenar(df)
            if not trained:
                raise RuntimeError("El modelo ML no pudo entrenar")
            core._state.update({
                "df": df,
                "elo_table": elo,
                "elo_map": elo_map,
                "ml": ml,
                "mc_context": load_mc_context(),
                "cache": cache_info(),
            })
    return core._state


def montecarlo_cached(*args, **kwargs):
    """Inyecta automáticamente el contexto Parquet sin cambiar la matemática MC."""
    if kwargs.get("mc_context") is None:
        ctx = core._state.get("mc_context")
        if ctx is not None:
            kwargs["mc_context"] = ctx
    return _mc_original(*args, **kwargs)


# Las rutas registradas en app_cloudrun_integrated resuelven estas funciones en
# el namespace del módulo en tiempo de ejecución, así que el reemplazo es seguro.
core.load_history = load_history_fast
core.model_state = optimized_model_state
core.simular_partido_montecarlo = montecarlo_cached
scanner_module.simular_partido_montecarlo = montecarlo_cached


# Mejora visual sin duplicar el HTML completo de la app integrada: cuando el
# scanner no encuentra picks, muestra también las razones de descarte y errores.
_OLD_SCAN_JS = "$('#scan').onclick=async()=>{let b=$('#scan');b.disabled=true;$('#scanstatus').textContent='Analizando los próximos 9 partidos...';$('#scanout').innerHTML='';try{let j=await api('/api/scan',{method:'POST'});$('#scanstatus').textContent=`${j.partidos_analizados} partido(s) analizados · ${j.picks.length} pick(s). ${j.sheet?.message||''}`;$('#scanout').innerHTML=pickCards(j.picks)}catch(e){$('#scanstatus').textContent=e.message}finally{b.disabled=false}}"
_NEW_SCAN_JS = "$('#scan').onclick=async()=>{let b=$('#scan');b.disabled=true;$('#scanstatus').textContent='Analizando los próximos 9 partidos...';$('#scanout').innerHTML='';try{let j=await api('/api/scan',{method:'POST'});let errs=(j.errores||[]).map(x=>({Estado:'ERROR',Motivo:x}));let diag=j.diagnostics||[];let extra=(diag.length||errs.length)?'<h3 style=\"margin-top:18px\">Diagnóstico del scanner</h3>'+table(diag.concat(errs)):'';$('#scanstatus').textContent=`${j.partidos_analizados} partido(s) analizados · ${j.picks.length} pick(s). ${j.sheet?.message||''}`;$('#scanout').innerHTML=pickCards(j.picks)+extra}catch(e){$('#scanstatus').textContent=e.message}finally{b.disabled=false}}"
if _OLD_SCAN_JS in core.HTML:
    core.HTML = core.HTML.replace(_OLD_SCAN_JS, _NEW_SCAN_JS)


def scan_with_diagnostics():
    try:
        s = core.model_state()
        df, em, ml = s["df"], s["elo_map"], s["ml"]
        upcoming = core.fixtures(force=True)[:core.SCANNER_MAX_FIXTURES]
        all_picks = []
        diagnostics = []
        errors = []

        for f in upcoming:
            partido = f"{f.get('local', '?')} vs {f.get('visita', '?')}"
            try:
                local, away, fid = f["local"], f["visita"], int(f["fixture_id"])
                if local not in em or away not in em:
                    msg = "sin ELO suficiente"
                    errors.append(f"{partido}: {msg}")
                    diagnostics.append({"Partido": partido, "Mercado": "Todos", "Estado": "NO BET", "Motivo": msg})
                    continue

                picks, diag = core.evaluar_fixture(
                    local, away, fid, df,
                    ml=ml, elo_map=em, return_diagnostics=True,
                )
                filtered = production_filter(picks)
                all_picks.extend(core.enrich_pick(p, f) for p in filtered)
                for d in diag:
                    row = dict(d)
                    row["Partido"] = partido
                    market = str(row.get("Mercado", ""))
                    if "Corners O/U" in market or "Tarjetas O/U" in market:
                        row["Estado"] = "ANÁLISIS SOLO"
                        row["Motivo"] = "Mercado temporalmente deshabilitado como pick por auditoría de calibración"
                    elif _is_quarter_line(market):
                        row["Estado"] = "NO BET"
                        row["Motivo"] = "Línea .25/.75 bloqueada hasta settlement asiático end-to-end"
                    diagnostics.append(row)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                errors.append(f"{partido}: {msg}")
                diagnostics.append({"Partido": partido, "Mercado": "Todos", "Estado": "ERROR", "Motivo": msg})

        all_picks.sort(
            key=lambda x: (float(x.get("EV_pct", 0) or 0), float(x.get("Edge_pp", 0) or 0)),
            reverse=True,
        )
        sheet_status = core.guardar_picks_ligamx(all_picks)
        return jsonify(core.safe({
            "partidos_analizados": len(upcoming),
            "limite_partidos": core.SCANNER_MAX_FIXTURES,
            "model_version": PRODUCTION_MODEL_VERSION,
            "picks": all_picks,
            "diagnostics": diagnostics,
            "sheet": sheet_status,
            "errores": errors[:20],
            "cache": s.get("cache", {}),
        }))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# Sustituye únicamente la view del endpoint ya registrado por la app integrada.
app.view_functions["scan"] = scan_with_diagnostics


@app.post("/api/settle")
def settle_ligamx_picks():
    try:
        return jsonify(liquidar_picks_pendientes())
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@app.get("/api/settle")
def settle_ligamx_picks_get():
    """Permite ejecución manual simple desde navegador/curl; es idempotente."""
    try:
        return jsonify(liquidar_picks_pendientes())
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@app.get("/health-v5")
def health_v5():
    return jsonify({
        "ok": True,
        "service": "liga-mx",
        "model_version": PRODUCTION_MODEL_VERSION,
        "policy": "1X2 + goles>=65%; corners/tarjetas analysis-only; edge<=20pp; no .25/.75",
    })
