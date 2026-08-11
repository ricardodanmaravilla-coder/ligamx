import os
import datetime
import pandas as pd
import requests

def generar_csv_cuotas_jornada():
    """Descarga la cartelera limpia de los próximos 9 partidos para la jornada."""
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard"
    
    hoy = datetime.date.today()
    futuro = hoy + datetime.timedelta(days=7)
    rango_fechas = f"{hoy.strftime('%Y%m%d')}-{futuro.strftime('%Y%m%d')}"
    
    params = {
        "limit": 9,  # Exactamente los 9 partidos de la jornada
        "dates": rango_fechas
    }
    
    partidos_data = []
    
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            events = data.get('events', [])[:9]
            
            for event in events:
                competencia = event.get('competitions', [{}])[0]
                competitors = competencia.get('competitors', [])
                local, visita = "", ""
                
                for comp in competitors:
                    team_name = comp.get('team', {}).get('displayName', '')
                    if comp.get('homeAway') == 'home':
                        local = team_name
                    else:
                        visita = team_name
                
                if local and visita:
                    partidos_data.append({
                        "Local": local.strip(),
                        "Visitante": visita.strip(),
                        "1": 0.00,
                        "X": 0.00,
                        "2": 0.00,
                        "Linea_Goles": 2.5,
                        "Over_Goles": 1.90,
                        "Linea_Corners": 9.5,
                        "Linea_Tarjetas": 4.5
                    })
                    
            if partidos_data:
                os.makedirs("data", exist_ok=True)
                df = pd.DataFrame(partidos_data)
                df.to_csv("data/cuotas_jornada.csv", index=False)
                print("✅ CSV base de la jornada generado limpiamente.")
                
    except Exception as e:
        print(f"⚠️ Error al generar el CSV: {e}")

if __name__ == "__main__":
    generar_csv_cuotas_jornada()
