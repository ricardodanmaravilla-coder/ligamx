import pandas as pd
import numpy as np

def analizar_apuestas_nfl(resultados, cuotas_personalizadas):
    """
    Analiza las probabilidades de la simulación de Montecarlo de la NFL
    contra las cuotas del usuario utilizando el Criterio de Kelly.
    """
    apuestas = []
    bankroll_pct = 0.02 # 2% por defecto para Kelly fraccional conservador

    # Mapeo de mercados simulados vs cuotas del usuario
    mercados = [
        {"mercado": f"Victoria Local", "prob_modelo": resultados['prob_local'], "llave": "1"},
        {"mercado": f"Victoria Visita", "prob_modelo": resultados['prob_visita'], "llave": "2"},
        {"mercado": "Over Total Puntos", "prob_modelo": resultados['prob_over'], "llave": "Over_Puntos"},
        {"mercado": "Under Total Puntos", "prob_modelo": resultados['prob_under'], "llave": "Under_Puntos"}
    ]

    for m in mercados:
        prob_m = m["prob_modelo"] / 100.0
        cuota = cuotas_personalizadas.get(m["llave"], 0.0)

        if prob_m > 0 and cuota > 1.0:
            prob_implícita = 1.0 / cuota
            ev = (prob_m * cuota) - 1.0

            # Criterio de Kelly
            kelly = (prob_m * cuota - 1.0) / (cuota - 1.0) if cuota > 1 else 0
            stake_recomendado = max(0.0, kelly * bankroll_pct * 100)

            # Lógica de Veredicto (Umbral del 70% + EV positivo)
            if m["prob_modelo"] >= 70.0 and ev > 0.05:
                veredicto = "🔥 APUESTA ESTRELLA"
            elif ev > 0.03:
                veredicto = "✅ Valor Detectado"
            elif ev > 0:
                veredicto = "⚠️ Valor Marginal"
            else:
                veredicto = "❌ Sin Valor (Evitar)"

            apuestas.append({
                "Mercado": m["mercado"],
                "Prob. Modelo": f"{m['prob_modelo']:.1f}%",
                "Cuota": f"{cuota:.2f}",
                "Prob. Implícita": f"{prob_implícita*100:.1f}%",
                "EV (%)": f"{ev*100:+.2f}%",
                "Stake (%)": f"{stake_recomendado:.2f}%",
                "Veredicto": veredicto
            })
        else:
            apuestas.append({
                "Mercado": m["mercado"],
                "Prob. Modelo": f"{m['prob_modelo']:.1f}%",
                "Cuota": f"{cuota:.2f}" if cuota > 0 else "N/D",
                "Prob. Implícita": "N/D",
                "EV (%)": "N/D",
                "Stake (%)": "0.00%",
                "Veredicto": "⏳ Ingresa Cuota"
            })

    return pd.DataFrame(apuestas)