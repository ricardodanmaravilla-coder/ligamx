import pandas as pd, numpy as n`
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from modules.ml_engine import PredictorML
from modules.feature_engineering import clean_history

def walk_forward(csv_path, min_train=800, step=150):
    df=clean_history(pd.read_csv(csv_path)); rows=[]
    # Retrain periodically and predict the next chronological block.
    for end in range(min_train,len(df),step):
        train=df.iloc[:end].copy(); test=df.iloc[end:min(end+step,len(df))].copy()
        ml=PredictorML()
        if not ml.entrenar(train): continue
        for _,r in test.iterrows():
            try:
                p=ml.predecir_mercados_completos(train,r.Local,r.Visitante,linea_goles=2.5,linea_corners=9.5,linea_tarjetas=4.5)
            except Exception: continue
            y=2 if r.Goles_L>r.Goles_V else 0 if r.Goles_L<r.Goles_V else 1
            rows.append({'Fecha':r.Fecha,'y':y,'pA':p['Resultado_1X2']['Gana Visita']/100,'pD':p['Resultado_1X2']['Empate']/100,'pH':p['Resultado_1X2']['Gana Local']/100,
                         'og25':p['Goles_Over_Under']['Over 2.5']/100,'yog25':int(r.Goles_L+r.Goles_V>2.5),
                         'oc95':p['Corners_Totales']['Over 9.5 Corners']/100,'yoc95':int(r.Corners_L+r.Corners_V>9.5),
                         'ot45':p['Tarjetas_Totales']['Over 4.5 Tarjetas']/100,'yot45':int(r.Amarillas_L+2*r.Rojas_L+r.Amarillas_V+2*r.Rojas_V>4.5)})
    out=pd.DataFrame(rows)
    if out.empty: return out,{}
    probs=out[['pA','pD','pH']].to_numpy(); y=out.y.to_numpy()
    metrics={'n':len(out),'logloss_1x2':log_loss(y,probs,labels=[0,1,2]),
             'brier_goals_o25':brier_score_loss(out.yog25,out.og25),
             'brier_corners_o95':brier_score_loss(out.yoc95,out.oc95),
             'brier_cards_o45':brier_score_loss(out.yot45,out.ot45)}
    return out,metrics

if __name__=='__main__':
    import sys
    rows,m=walk_forward(sys.argv[1] if len(sys.argv)>1 else 'historico_ligamx_completo.csv')
    print(m)
    rows.to_csv('backtest_predictions.csv',index=False)
