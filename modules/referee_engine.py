import pandas as pd

def calcular_perfiles_arbitros(df, min_matches=8, shrink=12):
    if df is None or 'Arbitro' not in df.columns: return {}
    d=df.copy(); d['Tarjetas_Totales']=d['Amarillas_L'].fillna(0)+2*d['Rojas_L'].fillna(0)+d['Amarillas_V'].fillna(0)+2*d['Rojas_V'].fillna(0)
    lg=float(d.Tarjetas_Totales.mean())
    if not lg: return {}
    out={}
    for ref,g in d.dropna(subset=['Arbitro']).groupby('Arbitro'):
        n=len(g)
        if n<min_matches: continue
        mean=(g.Tarjetas_Totales.sum()+shrink*lg)/(n+shrink)
        out[str(ref).strip()]=max(.85,min(1.15,mean/lg))
    return out

def obtener_factor_arbitro(nombre_arbitro, df=None):
    if not nombre_arbitro or df is None: return 1.0
    return calcular_perfiles_arbitros(df).get(str(nombre_arbitro).strip(),1.0)
