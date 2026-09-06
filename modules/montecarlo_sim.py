import numpy as np
from .stats_engine import calcular_expectativa_partido
from .feature_engineering import normalize_team


def _asian_side_probs(x, line, side):
    """Settlement probabilities for Asian totals.

    Returns full-win, half-win, push, half-loss and full-loss probabilities.
    Supports integer, half, quarter (.25) and three-quarter (.75) lines.
    """
    x = np.asarray(x)
    line = float(line)
    side = str(side).lower()
    frac = round(line - np.floor(line), 2)

    if frac in (0.25, 0.75):
        if frac == 0.25:
            n = int(np.floor(line))
            if side == "over":
                full_win = x >= n + 1
                half_win = np.zeros_like(x, dtype=bool)
                push = np.zeros_like(x, dtype=bool)
                half_loss = x == n
                full_loss = x <= n - 1
            else:
                full_win = x <= n - 1
                half_win = x == n
                push = np.zeros_like(x, dtype=bool)
                half_loss = np.zeros_like(x, dtype=bool)
                full_loss = x >= n + 1
        else:
            n = int(np.floor(line))
            split_int = n + 1
            if side == "over":
                full_win = x >= split_int + 1
                half_win = x == split_int
                push = np.zeros_like(x, dtype=bool)
                half_loss = np.zeros_like(x, dtype=bool)
                full_loss = x <= n
            else:
                full_win = x <= n
                half_win = np.zeros_like(x, dtype=bool)
                push = np.zeros_like(x, dtype=bool)
                half_loss = x == split_int
                full_loss = x >= split_int + 1
    else:
        if side == "over":
            full_win = x > line
            full_loss = x < line
        else:
            full_win = x < line
            full_loss = x > line
        push = x == line if line.is_integer() else np.zeros_like(x, dtype=bool)
        half_win = np.zeros_like(x, dtype=bool)
        half_loss = np.zeros_like(x, dtype=bool)

    pct = lambda mask: round(float(np.mean(mask) * 100.0), 1)
    return {
        "win": pct(full_win),
        "half_win": pct(half_win),
        "push": pct(push),
        "half_loss": pct(half_loss),
        "loss": pct(full_loss),
    }


def _market_probs(x, line):
    over = _asian_side_probs(x, line, "over")
    under = _asian_side_probs(x, line, "under")
    return over, under


def _market_payload(x, line, suffix=""):
    over, under = _market_probs(x, line)
    o = f"Over {line}{suffix}"
    u = f"Under {line}{suffix}"
    p = f"Push {line}{suffix}"
    return {
        o: over["win"],
        u: under["win"],
        p: max(over["push"], under["push"]),
        f"HalfWin {o}": over["half_win"],
        f"HalfLoss {o}": over["half_loss"],
        f"Loss {o}": over["loss"],
        f"HalfWin {u}": under["half_win"],
        f"HalfLoss {u}": under["half_loss"],
        f"Loss {u}": under["loss"],
    }


def simular_partido_montecarlo(local_raw, visita_raw, df_historico=None, elo_local=None, elo_visita=None, linea_goles=2.5, linea_corners=9.5, linea_tarjetas=4.5, num_simulaciones=250000, arbitro=None):
    local, visita = normalize_team(local_raw), normalize_team(visita_raw)
    e = calcular_expectativa_partido(local, visita, arbitro=arbitro, df=df_historico)
    gh, ga = e['lambda_goles_local'], e['lambda_goles_visita']
    if elo_local is not None and elo_visita is not None:
        d = max(-300, min(300, float(elo_local) - float(elo_visita)))
        gh *= np.exp(d / 2400)
        ga *= np.exp(-d / 2400)
    rng = np.random.default_rng()
    gl = rng.poisson(gh, num_simulaciones)
    gv = rng.poisson(ga, num_simulaciones)
    cl = rng.poisson(e['exp_corners_local'], num_simulaciones)
    cv = rng.poisson(e['exp_corners_visita'], num_simulaciones)
    tl = rng.poisson(e['exp_tarjetas_local'], num_simulaciones)
    tv = rng.poisson(e['exp_tarjetas_visita'], num_simulaciones)

    return {
        'Resultado_1X2': {
            'Gana Local': round(np.mean(gl > gv) * 100, 1),
            'Empate': round(np.mean(gl == gv) * 100, 1),
            'Gana Visita': round(np.mean(gl < gv) * 100, 1),
        },
        'Goles_Over_Under': _market_payload(gl + gv, linea_goles),
        'Corners_Totales': _market_payload(cl + cv, linea_corners, ' Corners'),
        'Tarjetas_Totales': _market_payload(tl + tv, linea_tarjetas, ' Tarjetas'),
        'Goles_Individuales': {local_raw: {'goles': round(gh, 2)}, visita_raw: {'goles': round(ga, 2)}},
        'Corners_Individuales': {local_raw: {'corners': round(e['exp_corners_local'], 2)}, visita_raw: {'corners': round(e['exp_corners_visita'], 2)}},
        'Tarjetas_Individuales': {local_raw: {'tarjetas': round(e['exp_tarjetas_local'], 2)}, visita_raw: {'tarjetas': round(e['exp_tarjetas_visita'], 2)}},
        'Contexto_Muestra': {'fallback_stats': bool(e.get('fallback_stats', False))},
    }
