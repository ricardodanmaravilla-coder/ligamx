import os
import numpy as np
import pandas as pd
from .feature_engineering import clean_history, normalize_team

ALTITUDES_LIGA_MX={'Toluca':2660,'CF Pachuca':2432,'U.N.A.M. - Pumas':2240,'Club America':2240,'Cruz Azul':2240,'Puebla':2135,'Club Tijuana':20,'Leon':1815,'Club Queretaro':1820,'Atletico San Luis':1850,'Necaxa':1888,'Atlas':1560,'Guadalajara Chivas':1560,'FC Juarez':1120,'Santos Laguna':1120,'Monterrey':540,'Tigres UANL':540,'Mazatlán':10}

def cargar_datos(path='data/historico_ligamx_completo.csv'):
    if not os.path.exists(path): path='historico_ligamx_completo.csv'
    return clean_history(pd.read_csv(path))

def _wavg(s, dates, half_life=240):
    age=(dates.max()-dates).dt.days
    w=0.5**(age/half_life)
    mask=s.notna() & w.notna()
    return float(np.average(s[mask],weights=w[mask])) if mask.any() else np.nan

def calcular_expectativa_partido(local, visitante, arbitro=None, df=None):
    df=clean_history(df if df is not None else cargar_datos())
    local,visitante=normalize_team(local),normalize_team(visitante)
    hl=df[df.Local==local].copy(); av=df[df.Visitante==visitante].copy()
    if len(hl)<8 or len(av)<8: raise ValueError('Muestra local/visitante insuficiente: NO BET')
    # Recent weighted goal rates. Real xG is used only where it is demonstrably not an exact-score fallback.
    league_h=_wavg(df.Goles_L,df.Fecha); league_a=_wavg(df.Goles_V,df.Fecha)
    ah=_wavg(hl.Goles_L,hl.Fecha)/league_h; dh=_wavg(hl.Goles_V,hl.Fecha)/league_a
    aa=_wavg(av.Goles_V,av.Fecha)/league_a; da=_wavg(av.Goles_L,av.Fecha)/league_h
    lam_h=max(.15, league_h*ah*da); lam_a=max(.15, league_a*aa*dh)
    # modest altitude adjustment only; not applied to cards/corners without fitted evidence
    delta=ALTITUDES_LIGA_MX.get(local,1500)-ALTITUDES_LIGA_MX.get(visitante,1500)
    if delta>=1200: lam_h*=1.04
    elif delta>=600: lam_h*=1.02
    # direct count expectations, no H2H multiplier
    c_h=(_wavg(hl.Corners_L,hl.Fecha)+_wavg(av.Corners_L,av.Fecha))/2
    c_a=(_wavg(av.Corners_V,av.Fecha)+_wavg(hl.Corners_V,hl.Fecha))/2
    card_h=(_wavg(hl.Amarillas_L+2*hl.Rojas_L,hl.Fecha)+_wavg(av.Amarillas_L+2*av.Rojas_L,av.Fecha))/2
    card_a=(_wavg(av.Amarillas_V+2*av.Rojas_V,av.Fecha)+_wavg(hl.Amarillas_V+2*hl.Rojas_V,hl.Fecha))/2
    if arbitro:
        try:
            from .referee_engine import obtener_factor_arbitro
            f=obtener_factor_arbitro(arbitro,df)
            card_h*=f; card_a*=f
        except Exception: pass
    return {'lambda_goles_local':lam_h,'lambda_goles_visita':lam_a,'exp_corners_local':max(.1,c_h),'exp_corners_visita':max(.1,c_a),'exp_tarjetas_local':max(.1,card_h),'exp_tarjetas_visita':max(.1,card_a)}
