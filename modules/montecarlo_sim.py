import numpy as np
from .stats_engine import calcular_expectativa_partido
from .feature_engineering import normalize_team

def _market_probs(x,line):
    line=float(line)
    over=float(np.mean(x>line)*100)
    push=float(np.mean(x==line)*100) if line.is_integer() else 0.0
    under=max(0.0,100.0-over-push)
    return round(over,1),round(under,1),round(push,1)

def simular_partido_montecarlo(local_raw, visita_raw, df_historico=None, elo_local=None, elo_visita=None, linea_goles=2.5, linea_corners=9.5, linea_tarjetas=4.5, num_simulaciones=250000, arbitro=None):
    local,visita=normalize_team(local_raw),normalize_team(visita_raw)
    e=calcular_expectativa_partido(local,visita,arbitro=arbitro,df=df_historico)
    gh,ga=e['lambda_goles_local'],e['lambda_goles_visita']
    if elo_local is not None and elo_visita is not None:
        d=max(-300,min(300,float(elo_local)-float(elo_visita)))
        gh*=np.exp(d/2400); ga*=np.exp(-d/2400)
    rng=np.random.default_rng()
    gl=rng.poisson(gh,num_simulaciones); gv=rng.poisson(ga,num_simulaciones)
    cl=rng.poisson(e['exp_corners_local'],num_simulaciones); cv=rng.poisson(e['exp_corners_visita'],num_simulaciones)
    tl=rng.poisson(e['exp_tarjetas_local'],num_simulaciones); tv=rng.poisson(e['exp_tarjetas_visita'],num_simulaciones)
    og,ug,pg=_market_probs(gl+gv,linea_goles); oc,uc,pc=_market_probs(cl+cv,linea_corners); ot,ut,pt=_market_probs(tl+tv,linea_tarjetas)
    return {'Resultado_1X2':{'Gana Local':round(np.mean(gl>gv)*100,1),'Empate':round(np.mean(gl==gv)*100,1),'Gana Visita':round(np.mean(gl<gv)*100,1)},
            'Goles_Over_Under':{f'Over {linea_goles}':og,f'Under {linea_goles}':ug,f'Push {linea_goles}':pg},
            'Corners_Totales':{f'Over {linea_corners} Corners':oc,f'Under {linea_corners} Corners':uc,f'Push {linea_corners} Corners':pc},
            'Tarjetas_Totales':{f'Over {linea_tarjetas} Tarjetas':ot,f'Under {linea_tarjetas} Tarjetas':ut,f'Push {linea_tarjetas} Tarjetas':pt},
            'Goles_Individuales':{local_raw:{'goles':round(gh,2)},visita_raw:{'goles':round(ga,2)}},
            'Corners_Individuales':{local_raw:{'corners':round(e['exp_corners_local'],2)},visita_raw:{'corners':round(e['exp_corners_visita'],2)}},
            'Tarjetas_Individuales':{local_raw:{'tarjetas':round(e['exp_tarjetas_local'],2)},visita_raw:{'tarjetas':round(e['exp_tarjetas_visita'],2)}}}
