import numpy as np
from .stats_engine import calcular_expectativa_partido
from .feature_engineering import normalize_team


def _market_probs(x, line):
    line = float(line)
    over = float(np.mean(x > line) * 100)
    push = float(np.mean(x == line) * 100) if line.is_integer() else 0.0
    under = max(0.0, 100.0 - over - push)
    return round(over, 1), round(under, 1), round(push, 1)


def _overdispersion_alpha(values):
    """Estima sobre-dispersión NB2 desde el histórico.

    Para NB2: Var(X) = mu + alpha * mu^2. Si no hay evidencia de
    sobre-dispersión, alpha=0 y se usa Poisson.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return 0.0
    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1))
    if mu <= 0 or var <= mu:
        return 0.0
    alpha = (var - mu) / (mu * mu)
    return float(np.clip(alpha, 0.0, 0.75))


def _draw_count(rng, mean_value, alpha, size):
    """Poisson si no hay sobre-dispersión; NegBinomial en caso contrario."""
    mu = max(0.01, float(mean_value))
    if alpha <= 1e-9:
        return rng.poisson(mu, size)
    n = 1.0 / float(alpha)
    p = n / (n + mu)
    return rng.negative_binomial(n, p, size)


def _historical_total(df, kind):
    if df is None:
        return np.array([])
    try:
        if kind == "corners":
            return (df["Corners_L"].astype(float) + df["Corners_V"].astype(float)).to_numpy()
        if kind == "tarjetas":
            return (
                df["Amarillas_L"].astype(float) + 2.0 * df["Rojas_L"].astype(float)
                + df["Amarillas_V"].astype(float) + 2.0 * df["Rojas_V"].astype(float)
            ).to_numpy()
    except Exception:
        return np.array([])
    return np.array([])


def simular_partido_montecarlo(
    local_raw, visita_raw, df_historico=None, elo_local=None, elo_visita=None,
    linea_goles=2.5, linea_corners=9.5, linea_tarjetas=4.5,
    num_simulaciones=250000, arbitro=None
):
    local, visita = normalize_team(local_raw), normalize_team(visita_raw)
    e = calcular_expectativa_partido(local, visita, arbitro=arbitro, df=df_historico)
    gh, ga = e["lambda_goles_local"], e["lambda_goles_visita"]

    if elo_local is not None and elo_visita is not None:
        d = max(-300, min(300, float(elo_local) - float(elo_visita)))
        gh *= np.exp(d / 2400)
        ga *= np.exp(-d / 2400)

    rng = np.random.default_rng()

    # Goles: Poisson sigue siendo la base por equipo para conservar 1X2.
    gl = rng.poisson(gh, num_simulaciones)
    gv = rng.poisson(ga, num_simulaciones)

    # Corners/tarjetas: usamos la dispersión real observada del histórico. El modelo
    # anterior forzaba Poisson y podía subestimar las colas, inflando probabilidades
    # como 75-80% en líneas altas/bajas.
    alpha_c = _overdispersion_alpha(_historical_total(df_historico, "corners"))
    alpha_t = _overdispersion_alpha(_historical_total(df_historico, "tarjetas"))
    corners_total = _draw_count(
        rng, e["exp_corners_local"] + e["exp_corners_visita"], alpha_c, num_simulaciones
    )
    cards_total = _draw_count(
        rng, e["exp_tarjetas_local"] + e["exp_tarjetas_visita"], alpha_t, num_simulaciones
    )

    og, ug, pg = _market_probs(gl + gv, linea_goles)
    oc, uc, pc = _market_probs(corners_total, linea_corners)
    ot, ut, pt = _market_probs(cards_total, linea_tarjetas)

    return {
        "Resultado_1X2": {
            "Gana Local": round(np.mean(gl > gv) * 100, 1),
            "Empate": round(np.mean(gl == gv) * 100, 1),
            "Gana Visita": round(np.mean(gl < gv) * 100, 1),
        },
        "Goles_Over_Under": {
            f"Over {linea_goles}": og,
            f"Under {linea_goles}": ug,
            f"Push {linea_goles}": pg,
        },
        "Corners_Totales": {
            f"Over {linea_corners} Corners": oc,
            f"Under {linea_corners} Corners": uc,
            f"Push {linea_corners} Corners": pc,
        },
        "Tarjetas_Totales": {
            f"Over {linea_tarjetas} Tarjetas": ot,
            f"Under {linea_tarjetas} Tarjetas": ut,
            f"Push {linea_tarjetas} Tarjetas": pt,
        },
        "Goles_Individuales": {
            local_raw: {"goles": round(gh, 2)},
            visita_raw: {"goles": round(ga, 2)},
        },
        "Corners_Individuales": {
            local_raw: {"corners": round(e["exp_corners_local"], 2)},
            visita_raw: {"corners": round(e["exp_corners_visita"], 2)},
        },
        "Tarjetas_Individuales": {
            local_raw: {"tarjetas": round(e["exp_tarjetas_local"], 2)},
            visita_raw: {"tarjetas": round(e["exp_tarjetas_visita"], 2)},
        },
        "Dispersion": {
            "corners_alpha": round(alpha_c, 4),
            "tarjetas_alpha": round(alpha_t, 4),
        },
    }
