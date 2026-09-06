import numpy as np
import pandas as pd

TEAM_ALIASES = {
    'América': 'Club America', 'America': 'Club America', 'Club América': 'Club America',
    'Pachuca': 'CF Pachuca',
    'Pumas': 'U.N.A.M. - Pumas', 'Pumas UNAM': 'U.N.A.M. - Pumas', 'UNAM': 'U.N.A.M. - Pumas',
    'Guadalajara': 'Guadalajara Chivas', 'Chivas': 'Guadalajara Chivas', 'Chivas Guadalajara': 'Guadalajara Chivas',
    'Juárez': 'FC Juarez', 'Juarez': 'FC Juarez',
    'Querétaro': 'Club Queretaro', 'Queretaro': 'Club Queretaro',
    'León': 'Leon',
    'Atlético de San Luis': 'Atletico San Luis', 'Atlético San Luis': 'Atletico San Luis', 'San Luis': 'Atletico San Luis',
    'Tijuana': 'Club Tijuana',
    'Tigres': 'Tigres UANL', 'U.A.N.L.': 'Tigres UANL',
    'Santos': 'Santos Laguna',
    'Mazatlan': 'Mazatlán', 'Mazatlan FC': 'Mazatlán',
}

NUMERIC_COLS = [
    'Goles_L','Goles_V','Corners_L','Corners_V','Amarillas_L','Amarillas_V','Rojas_L','Rojas_V',
    'xG_L','xG_V','Atajadas_L','Atajadas_V','Tiros_Al_Arco_L','Tiros_Al_Arco_V'
]
METRICS = ['GF','GA','CF','CA','CardsF','CardsA','SOTF','SOTA','Pts','SavePct']


def normalize_team(name):
    if not isinstance(name, str):
        return name
    name = name.strip()
    return TEAM_ALIASES.get(name, name)


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha','Local','Visitante']).sort_values('Fecha').reset_index(drop=True)
    df['Local'] = df['Local'].map(normalize_team)
    df['Visitante'] = df['Visitante'].map(normalize_team)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def add_pre_match_elo(df: pd.DataFrame, k=24.0, home_adv=45.0, base=1500.0) -> pd.DataFrame:
    df = clean_history(df)
    ratings = {}
    el, ev = [], []
    for _, r in df.iterrows():
        h, a = r['Local'], r['Visitante']
        rh, ra = ratings.get(h, base), ratings.get(a, base)
        el.append(rh)
        ev.append(ra)
        exp_h = 1.0 / (1.0 + 10.0 ** ((ra - (rh + home_adv)) / 400.0))
        if r['Goles_L'] > r['Goles_V']:
            sh = 1.0
        elif r['Goles_L'] < r['Goles_V']:
            sh = 0.0
        else:
            sh = 0.5
        margin = abs(float(r['Goles_L']) - float(r['Goles_V']))
        mult = 1.0 + 0.15 * np.log1p(margin)
        delta = k * mult * (sh - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    df['ELO_Local_Pre'] = el
    df['ELO_Visita_Pre'] = ev
    df['Diff_ELO_Pre'] = df['ELO_Local_Pre'] - df['ELO_Visita_Pre']
    return df


def _team_long(df):
    home = pd.DataFrame({
        'idx': df.index, 'Fecha': df['Fecha'], 'Equipo': df['Local'], 'Rival': df['Visitante'], 'Es_Local': 1,
        'GF': df['Goles_L'], 'GA': df['Goles_V'], 'CF': df['Corners_L'], 'CA': df['Corners_V'],
        'CardsF': df['Amarillas_L'] + 2 * df['Rojas_L'], 'CardsA': df['Amarillas_V'] + 2 * df['Rojas_V'],
        'SOTF': df['Tiros_Al_Arco_L'], 'SOTA': df['Tiros_Al_Arco_V'], 'Saves': df['Atajadas_L'],
    })
    away = pd.DataFrame({
        'idx': df.index, 'Fecha': df['Fecha'], 'Equipo': df['Visitante'], 'Rival': df['Local'], 'Es_Local': 0,
        'GF': df['Goles_V'], 'GA': df['Goles_L'], 'CF': df['Corners_V'], 'CA': df['Corners_L'],
        'CardsF': df['Amarillas_V'] + 2 * df['Rojas_V'], 'CardsA': df['Amarillas_L'] + 2 * df['Rojas_L'],
        'SOTF': df['Tiros_Al_Arco_V'], 'SOTA': df['Tiros_Al_Arco_L'], 'Saves': df['Atajadas_V'],
    })
    long = pd.concat([home, away], ignore_index=True).sort_values(['Equipo','Fecha','idx'])
    long['Pts'] = np.where(long.GF > long.GA, 3, np.where(long.GF == long.GA, 1, 0))
    long['SavePct'] = long['Saves'] / (long['Saves'] + long['GA']).replace(0, np.nan)
    return long


def _shifted_ewm(series, span, min_periods):
    return series.shift(1).ewm(span=span, adjust=False, min_periods=min_periods).mean()


def add_rolling_features(df: pd.DataFrame, windows=(5, 10), ewm_spans=(5, 10)) -> pd.DataFrame:
    df = add_pre_match_elo(df)
    long = _team_long(df)

    for w in windows:
        for m in METRICS:
            long[f'{m}_{w}'] = long.groupby('Equipo')[m].transform(
                lambda s: s.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
            )
        long[f'MatchesBefore_{w}'] = long.groupby('Equipo').cumcount()

    for span in ewm_spans:
        minp = max(2, span // 2)
        for m in METRICS:
            long[f'{m}_EWM{span}'] = long.groupby('Equipo')[m].transform(
                lambda s, sp=span, mp=minp: _shifted_ewm(s, sp, mp)
            )

    keep = ['idx','Equipo','Es_Local'] + [
        c for c in long.columns
        if any(c.endswith(f'_{w}') for w in windows)
        or any(c.endswith(f'_EWM{span}') for span in ewm_spans)
        or c.startswith('MatchesBefore_')
    ]
    h = long[long.Es_Local == 1][keep].drop(columns=['Es_Local','Equipo']).set_index('idx').add_prefix('H_')
    a = long[long.Es_Local == 0][keep].drop(columns=['Es_Local','Equipo']).set_index('idx').add_prefix('A_')
    out = df.join(h).join(a)

    for span in ewm_spans:
        for m in METRICS:
            hc, ac = f'H_{m}_EWM{span}', f'A_{m}_EWM{span}'
            if hc in out.columns and ac in out.columns:
                out[f'Diff_{m}_EWM{span}'] = out[hc] - out[ac]

    if 'xG_L' in out.columns and 'xG_V' in out.columns:
        out['xG_real_flag'] = (
            ((out['xG_L'] - out['Goles_L']).abs() > 1e-9) |
            ((out['xG_V'] - out['Goles_V']).abs() > 1e-9)
        ).astype(int)
    else:
        out['xG_real_flag'] = 0
    return out


def build_current_team_feature_cache(df, windows=(5, 10), ewm_spans=(5, 10)):
    """Precalcula una sola vez la forma vigente de cada equipo para el scanner."""
    long = _team_long(clean_history(df))
    cache = {}
    for team, g in long.groupby('Equipo', sort=False):
        g = g.sort_values(['Fecha', 'idx'])
        snap = {}
        for w in windows:
            minp = max(2, w // 2)
            for m in METRICS:
                s = g[m].dropna()
                snap[f'{m}_{w}'] = float(s.tail(w).mean()) if len(s) >= minp else np.nan
            snap[f'MatchesBefore_{w}'] = int(len(g))
        for span in ewm_spans:
            minp = max(2, span // 2)
            for m in METRICS:
                s = g[m]
                if s.notna().sum() >= minp:
                    val = s.ewm(span=span, adjust=False, min_periods=minp).mean().iloc[-1]
                    snap[f'{m}_EWM{span}'] = float(val) if pd.notna(val) else np.nan
                else:
                    snap[f'{m}_EWM{span}'] = np.nan
        cache[team] = snap
    return cache


def current_match_features(df, home, away, min_matches=5, team_cache=None,
                           elo_local=None, elo_visita=None, windows=(5, 10), ewm_spans=(5, 10)):
    home, away = normalize_team(home), normalize_team(away)

    if team_cache is not None and elo_local is not None and elo_visita is not None:
        if home not in team_cache or away not in team_cache:
            raise ValueError(f'Equipo desconocido: {home if home not in team_cache else away}')
        hs, as_ = team_cache[home], team_cache[away]
        if hs.get('MatchesBefore_5', 0) < min_matches or as_.get('MatchesBefore_5', 0) < min_matches:
            raise ValueError('Muestra historica insuficiente para uno de los equipos')
        feat = {
            'ELO_Local_Pre': float(elo_local),
            'ELO_Visita_Pre': float(elo_visita),
            'Diff_ELO_Pre': float(elo_local) - float(elo_visita),
        }
        for k, v in hs.items():
            feat[f'H_{k}'] = v
        for k, v in as_.items():
            feat[f'A_{k}'] = v
        for span in ewm_spans:
            for m in METRICS:
                hv, av = hs.get(f'{m}_EWM{span}'), as_.get(f'{m}_EWM{span}')
                feat[f'Diff_{m}_EWM{span}'] = (
                    float(hv) - float(av)
                    if hv is not None and av is not None and pd.notna(hv) and pd.notna(av)
                    else np.nan
                )
        return pd.Series(feat)

    hist = add_rolling_features(df, windows=windows, ewm_spans=ewm_spans)
    teams = set(hist['Local']) | set(hist['Visitante'])
    if home not in teams or away not in teams:
        raise ValueError(f'Equipo desconocido: {home if home not in teams else away}')

    future_date = hist['Fecha'].max() + pd.Timedelta(days=1)
    row = {c: np.nan for c in clean_history(df).columns}
    row.update({
        'Fecha': future_date, 'Local': home, 'Visitante': away,
        'Goles_L': 0, 'Goles_V': 0, 'Corners_L': 0, 'Corners_V': 0,
        'Amarillas_L': 0, 'Amarillas_V': 0, 'Rojas_L': 0, 'Rojas_V': 0,
        'xG_L': 0, 'xG_V': 0, 'Atajadas_L': 0, 'Atajadas_V': 0,
        'Tiros_Al_Arco_L': 0, 'Tiros_Al_Arco_V': 0, 'Arbitro': ''
    })
    tmp = pd.concat([clean_history(df), pd.DataFrame([row])], ignore_index=True)
    feat = add_rolling_features(tmp, windows=windows, ewm_spans=ewm_spans).iloc[-1]
    if feat.get('H_MatchesBefore_5', 0) < min_matches or feat.get('A_MatchesBefore_5', 0) < min_matches:
        raise ValueError('Muestra historica insuficiente para uno de los equipos')
    return feat
