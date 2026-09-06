import numpy as np
import pandas as pd

from .feature_engineering import clean_history, normalize_team
from .stats_engine import _context_rate, _team_overall, _wavg, ALTITUDES_LIGA_MX


def build_mc_context(df):
    """Precalcula los inputs estadísticos que Monte Carlo reutiliza por equipo."""
    df = clean_history(df)

    league_h = _wavg(df.Goles_L, df.Fecha)
    league_a = _wavg(df.Goles_V, df.Fecha)
    league_team_goals = (league_h + league_a) / 2.0

    league_ch = _wavg(df.Corners_L, df.Fecha)
    league_ca = _wavg(df.Corners_V, df.Fecha)
    league_ct = (league_ch + league_ca) / 2.0

    league_card_h = _wavg(df.Amarillas_L + 2 * df.Rojas_L, df.Fecha)
    league_card_a = _wavg(df.Amarillas_V + 2 * df.Rojas_V, df.Fecha)
    league_card_t = (league_card_h + league_card_a) / 2.0

    teams = sorted(set(df.Local) | set(df.Visitante))
    rows = []
    for team in teams:
        hl = df[df.Local == team]
        av = df[df.Visitante == team]
        overall = _team_overall(df, team)
        if len(overall) < 5:
            continue

        def cr(spec, spec_dates, spec_mean, overall_col, overall_mean):
            return _context_rate(
                spec, spec_dates, spec_mean,
                overall[overall_col], overall.Fecha, overall_mean,
            )

        rows.append({
            'Equipo': team,
            'home_gf': cr(hl.Goles_L, hl.Fecha, league_h, 'GF', league_team_goals),
            'home_ga': cr(hl.Goles_V, hl.Fecha, league_a, 'GA', league_team_goals),
            'away_gf': cr(av.Goles_V, av.Fecha, league_a, 'GF', league_team_goals),
            'away_ga': cr(av.Goles_L, av.Fecha, league_h, 'GA', league_team_goals),
            'home_cf': cr(hl.Corners_L, hl.Fecha, league_ch, 'CF', league_ct),
            'home_ca': cr(hl.Corners_V, hl.Fecha, league_ca, 'CA', league_ct),
            'away_cf': cr(av.Corners_V, av.Fecha, league_ca, 'CF', league_ct),
            'away_ca': cr(av.Corners_L, av.Fecha, league_ch, 'CA', league_ct),
            'home_cards_f': cr(hl.Amarillas_L + 2 * hl.Rojas_L, hl.Fecha, league_card_h, 'CardsF', league_card_t),
            'home_cards_a': cr(hl.Amarillas_V + 2 * hl.Rojas_V, hl.Fecha, league_card_a, 'CardsA', league_card_t),
            'away_cards_f': cr(av.Amarillas_V + 2 * av.Rojas_V, av.Fecha, league_card_a, 'CardsF', league_card_t),
            'away_cards_a': cr(av.Amarillas_L + 2 * av.Rojas_L, av.Fecha, league_card_h, 'CardsA', league_card_t),
            'n_home': int(len(hl)),
            'n_away': int(len(av)),
            'n_total': int(len(overall)),
            'league_goles_h': float(league_h),
            'league_goles_a': float(league_a),
        })
    return pd.DataFrame(rows)


def context_to_map(context_df):
    if context_df is None or context_df.empty:
        return None
    return {str(r['Equipo']): r for r in context_df.to_dict(orient='records')}


def calcular_expectativa_contexto(local, visitante, context_map, arbitro=None, df=None):
    local, visitante = normalize_team(local), normalize_team(visitante)
    if not context_map or local not in context_map or visitante not in context_map:
        raise ValueError('Muestra histórica total insuficiente (<5 partidos): NO BET')

    h = context_map[local]
    a = context_map[visitante]
    league_h = max(float(h['league_goles_h']), 1e-6)
    league_a = max(float(h['league_goles_a']), 1e-6)

    ah = float(h['home_gf']) / league_h
    dh = float(h['home_ga']) / league_a
    aa = float(a['away_gf']) / league_a
    da = float(a['away_ga']) / league_h
    lam_h = max(.15, league_h * ah * da)
    lam_a = max(.15, league_a * aa * dh)

    delta = ALTITUDES_LIGA_MX.get(local, 1500) - ALTITUDES_LIGA_MX.get(visitante, 1500)
    if delta >= 1200:
        lam_h *= 1.04
    elif delta >= 600:
        lam_h *= 1.02

    c_h = (float(h['home_cf']) + float(a['away_ca'])) / 2.0
    c_a = (float(a['away_cf']) + float(h['home_ca'])) / 2.0
    card_h = (float(h['home_cards_f']) + float(a['away_cards_a'])) / 2.0
    card_a = (float(a['away_cards_f']) + float(h['home_cards_a'])) / 2.0

    if arbitro and df is not None:
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
        'fallback_stats': int(h['n_home']) < 8 or int(a['n_away']) < 8,
        'muestra_local_sede': int(h['n_home']),
        'muestra_visita_sede': int(a['n_away']),
        'muestra_local_total': int(h['n_total']),
        'muestra_visita_total': int(a['n_total']),
    }
