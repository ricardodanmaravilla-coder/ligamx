from flask import jsonify

from app_cloudrun_integrated import app
from modules.settler_ligamx import liquidar_picks_pendientes


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
