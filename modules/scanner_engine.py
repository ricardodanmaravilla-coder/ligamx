import os, pandas as pd
from .feature_engineering import clean_history, normalize_team
from .elo_engine import SistemaEloLigaMX
from .ml_engine import PredictorML
from .montecarlo_sim import simular_partido_montecarlo
from .odds_engine import obtener_cuotas_partido, remove_vig_two_way

def combinar_probabilidades(p_stat,p_ml,w_stat=.55,w_ml=.45):
    # Ensemble, not "independent consensus". Weights must later be tuned by walk-forward backtest.
    return w_stat*float(p_stat)+w_ml*float(p_ml)

def evaluar_fixture(local,visita,fixture_id,df_historico,cuotas=None,min_prob=55,min_edge=3,min_ev=3,max_disagreement=12):
    df=clean_history(df_historico); local=normalize_team(local); visita=normalize_team(visita)
    elo=SistemaEloLigaMX(); table=elo.calcular_historico(df); em=dict(zip(table.Equipo,table.ELO_Rating))
    if local not in em or visita not in em: return []
    cuotas=cuotas or obtener_cuotas_partido(fixture_id)
    if not cuotas: return []
    ml=PredictorML()
    if not ml.entrenar(df): return []
    lg=cuotas.get('linea_goles_detectada'); lc=cuotas.get('linea_corners_detectada'); lt=cuotas.get('linea_tarjetas_detectada')
    # Markets with no verified real line simply are not evaluated.
    mc=simular_partido_montecarlo(local,visita,df_historico=df,elo_local=em[local],elo_visita=em[visita],linea_goles=float(lg) if lg else 2.5,linea_corners=float(lc) if lc else 9.5,linea_tarjetas=float(lt) if lt else 4.5)
    mlp=ml.predecir_mercados_completos(df,local,visita,elo_local=em[local],elo_visita=em[visita],linea_goles=float(lg) if lg else 2.5,linea_corners=float(lc) if lc else 9.5,linea_tarjetas=float(lt) if lt else 4.5)
    markets=[('Gana Local','1',mc['Resultado_1X2']['Gana Local'],mlp['Resultado_1X2']['Gana Local']),('Empate','X',mc['Resultado_1X2']['Empate'],mlp['Resultado_1X2']['Empate']),('Gana Visita','2',mc['Resultado_1X2']['Gana Visita'],mlp['Resultado_1X2']['Gana Visita'])]
    if lg:
        markets += [(f'Over {lg} Goles',f'Over {lg}',mc['Goles_Over_Under'][f'Over {float(lg)}'],mlp['Goles_Over_Under'][f'Over {float(lg)}']),
                    (f'Under {lg} Goles',f'Under {lg}',mc['Goles_Over_Under'][f'Under {float(lg)}'],mlp['Goles_Over_Under'][f'Under {float(lg)}'])]
    if lc:
        markets += [(f'Over {lc} Corners',f'Over {lc} Corners',mc['Corners_Totales'][f'Over {float(lc)} Corners'],mlp['Corners_Totales'][f'Over {float(lc)} Corners']),
                    (f'Under {lc} Corners',f'Under {lc} Corners',mc['Corners_Totales'][f'Under {float(lc)} Corners'],mlp['Corners_Totales'][f'Under {float(lc)} Corners'])]
    if lt:
        markets += [(f'Over {lt} Tarjetas',f'Over {lt} Tarjetas',mc['Tarjetas_Totales'][f'Over {float(lt)} Tarjetas'],mlp['Tarjetas_Totales'][f'Over {float(lt)} Tarjetas']),
                    (f'Under {lt} Tarjetas',f'Under {lt} Tarjetas',mc['Tarjetas_Totales'][f'Under {float(lt)} Tarjetas'],mlp['Tarjetas_Totales'][f'Under {float(lt)} Tarjetas'])]
    out=[]
    for name,key,p1,p2 in markets:
        odd=cuotas.get(key)
        if not odd or float(odd)<=1.01: continue
        if abs(float(p1)-float(p2))>max_disagreement: continue
        pe=combinar_probabilidades(p1,p2)
        market_p=100/float(odd) # no-vig replacement below for two-way markets
        if name.startswith(('Over','Under')):
            opp=('Under '+key[5:]) if key.startswith('Over ') else ('Over '+key[6:])
            if cuotas.get(opp):
                a,b=remove_vig_two_way(odd,cuotas[opp]); market_p=a
        edge=pe-market_p; ev=(pe/100*float(odd)-1)*100
        if pe>=min_prob and edge>=min_edge and ev>=min_ev:
            out.append({'Partido':f'{local} vs {visita}','Mercado':name,'P_Estadistico':round(float(p1),1),'P_ML':round(float(p2),1),'P_Ensemble':round(pe,1),'Cuota':round(float(odd),2),'P_Mercado':round(market_p,1),'Edge_pp':round(edge,1),'EV_pct':round(ev,1),'Veredicto':'VALUE BET'})
    return sorted(out,key=lambda x:(x['EV_pct'],x['Edge_pp']),reverse=True)
