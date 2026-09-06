import os
import numpy as np
import pandas as pd
from .feature_engineering import clean_history, normalize_team

ALTITUDES_LIGA_MX = {
    'Toluca': 2660, 'CF Pachuca': 2432, 'U.N.A.M. - Pumas': 2240,
    'Club America': 2240, 'Cruz Azul': 2240, 'Puebla': 2135,
    'Club Tijuana': 20, 'Leon': 1815, 'Club Queretaro': 1820,
    'Atletico San Luis': 1850, 'Necaxa': 1888, 'Atlas': 1560,
    'Guadalajara Chivas': 1560, 'FC Juarez': 1120, 'Santos Laguna': 1120,
    'Monterrey': 540, 'Tigres UANL': 540, 'Mazatlán': 10,
}


def cargar_datos(path='data/historico_ligamx_completo.csv'):
    if not os.path.exists(path):
        path = 'historico_ligamx_completo.csv'
    return clean_history(pd.read_csv(path))


def _wavg(s, dates, half_life=240):
    age = (dates.max() - dates).dt.days
    w = 0.5 ** (age / half_life)
    mask = s.notna() & w.notna()
    return float(np.average(s[mask], weights=w[mask])) if mask.any() else np.nan


def _team_overall(df, team):
    h = df[df.Local == team]
    a = df[df.Visitante == team]
    rows = []
    if not h.empty:
        rows.append(pd.DataFrame({
            'Fecha': h.Fecha,
            'GF': h.Goles_L, 'GA': h.Goles_V,
            'CF': h.Corners_L, 'CA': h.Corners_V,
            'CardsF': h.Amarillas_L + 2 * h.Rojas_L,
            'CardsA': h.Amarillas_V + 2 * h.Rojas_V,
        }))
    if not a.empty:
        rows.append(pd.DataFrame({
            'Fecha': a.Fecha,
            'GF': a.Goles_V, 'GA': a.Goles_L,
            'CF': a.Corners_V, 'CA': a.Corners_L,
            'CardsF': a.Amarillas_V + 2 * a.Rojas_V,
            'CardsA': a.Amarillas_L + 2 * a.Rojas_L,
        }))
    if not rows:
        return pd.DataFrame(columns=['Fecha','GF','GA','CF','CA','CardsF','CardsA'])
    return pd.concat(rows, ignore_index=True).sort_values('Fecha')


def _context_rate(specific, specific_dates, specific_league_mean,
                  overall, overall_dates, overall_league_mean,
                  target_specific=8, target_overall=12):
    """Shrink a short venue split toward team overall form and league average."""
    specific_league_mean = max(float(specific_league_mean), 1e-6)
    overall_league_mean = max(float(overall_league_mean), 1e-6)

    o_est = _wavg(overall, overall_dates) if len(overall) else np.nan
    if not np.isfinite(o_est):
        o_est = overall_league_mean
    overall_n = int(overall.notna().sum()) if hasattr(overall, 'notna') else len(overall)
    a_overall = min(1.0, overall_n / float(target_overall))
    overall_ratio = o_est / overall_league_mean
    shrunk_overall_ratio = a_overall * overall_ratio + (1.0 - a_overall) * 1.0

    s_est = _wavg(specific, specific_dates) if len(specific) else np.nan
    specific_n = int(specific.notna().sum()) if hasattr(specific, 'notna') else len(specific)
    if np.isfinite(s_est) and specific_n > 0:
        specific_ratio = s_est / specific_league_mean
        a_specific = min(1.0, specific_n / float(target_specific))
        ratio = a_specific * specific_ratio + (1.0 - a_specific) * shrunk_overall_ratio
    else:
        ratio = shrunk_overall_ratio
    return max(0.05, specific_league_mean * ratio)


def calcular_expectativa_partido(local, visitante, arbitro=None, df=None):
    df = clean_history(df if df is not None else cargar_datos())
    local, visitante = normalize_team(local), normalize_team(visitante)
    hl = df[df.Local == local].copy()
    av = df[df.Visitante == visitante].copy()
    local_all = _team_overall(df, local)
    visit_all = _team_overall(df, visitante)

    # We still require a minimal real sample. The fallback is not allowed to invent
    # a completely unknown team's strength from league averages alone.
    if len(local_all) < 5 or len(visit_all) < 5:
        raise ValueError('Muestra histórica total insuficiente (<5 partidos): NO BET')

    fallback_stats = len(hl) < 8 or len(av) < 8

    league_h = _wavg(df.Goles_L, df.Fecha)
    league_a = _wavg(df.Goles_V, df.Fecha)
    league_team_goals = (league_h + league_a) / 2.0

    # Attack/defence rates are estimated in the correct venue context; when the
    # split is short, they are shrunk toward overall team form and then league mean.
    local_gf = _context_rate(
        hl.Goles_L, hl.Fecha, league_h,
        local_all.GF, local_all.Fecha, league_team_goals,
    )
    local_ga = _context_rate(
        hl.Goles_V, hl.Fecha, league_a,
        local_all.GA, local_all.Fecha, league_team_goals,
    )
    visit_gf = _context_rate(
        av.Goles_V, av.Fecha, league_a,
        visit_all.GF, visit_all.Fecha, league_team_goals,
    )
    visit_ga = _context_rate(
        av.Goles_L, av.Fecha, league_h,
        visit_all.GA, visit_all.Fecha, league_team_goals,
    )

    ah = local_gf / league_h
    dh = local_ga / league_a
    aa = visit_gf / league_a
    da = visit_ga / league_h
    lam_h = max(.15, league_h * ah * da)
    lam_a = max(.15, league_a * aa * dh)

    delta = ALTITUDES_LIGA_MX.get(local, 1500) - ALTITUDES_LIGA_MX.get(visitante, 1500)
    if delta >= 1200:
        lam_h *= 1.04
    elif delta >= 600:
        lam_h *= 1.02

    league_ch = _wavg(df.Corners_L, df.Fecha)
    league_ca = _wavg(df.Corners_V, df.Fecha)
    league_ct = (league_ch + league_ca) / 2.0
    local_cf = _context_rate(hl.Corners_L, hl.Fecha, league_ch, local_all.CF, local_all.Fecha, league_ct)
    local_ca = _context_rate(hl.Corners_V, hl.Fecha, league_ca, local_all.CA, local_all.Fecha, league_ct)
    visit_cf = _context_rate(av.Corners_V, av.Fecha, league_ca, visit_all.CF, visit_all.Fecha, league_ct)
    visit_ca = _context_rate(av.Corners_L, av.Fecha, league_ch, visit_all.CA, visit_all.Fecha, league_ct)
    c_h = (local_cf + visit_ca) / 2.0
    c_a = (visit_cf + local_ca) / 2.0

    league_card_h = _wavg(df.Amarillas_L + 2 * df.Rojas_L, df.Fecha)
    league_card_a = _wavg(df.Amarillas_V + 2 * df.Rojas_V, df.Fecha)
    league_card_t = (league_card_h + league_card_a) / 2.0
    local_cards_f = _context_rate(
        hl.Amarillas_L + 2 * hl.Rojas_L, hl.Fecha, league_card_h,
        local_all.CardsF, local_all.Fecha, league_card_t,
    )
    local_cards_a = _context_rate(
        hl.Amarillas_V + 2 * hl.Rojas_V, hl.Fecha, league_card_a,
        local_all.CardsA, local_all.Fecha, league_card_t,
    )
    visit_cards_f = _context_rate(
        av.Amarillas_V + 2 * av.Rojas_V, av.Fecha, league_card_a,
        visit_all.CardsF, visit_all.Fecha, league_card_t,
    )
    visit_cards_a = _context_rate(
        av.Amarillas_L + 2 * av.Rojas_L, av.Fecha, league_card_h,
        visit_all.CardsA, visit_all.Fecha, league_card_t,
    )
    card_h = (local_cards_f + visit_cards_a) / 2.0
    card_a = (visit_cards_f + local_cards_a) / 2.0

    if arbitro:
        try:
            from .referee_engine import obtener_factor_arbitro
            f = obtener_factor_arbitro(arbitro, df)
            card_h *= f
            card_a *= f
        except Exception:
            pass

    return {
        'lambda_goles_local': lam_h,
        'lambda_goles_visita': lam_a,
        'exp_corners_local': max(.1, c_h),
        'exp_corners_visita': max(.1, c_a),
        'exp_tarjetas_local': max(.1, card_h),
        'exp_tarjetas_visita': max(.1, card_a),
        'fallback_stats': fallback_stats,
        'muestra_local_sede': int(len(hl)),
        'muestra_visita_sede': int(len(av)),
        'muestra_local_total': int(len(local_all)),
        'muestra_visita_total': int(len(visit_all)),
    }
