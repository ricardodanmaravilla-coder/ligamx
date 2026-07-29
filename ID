import os
import requests
import json

# Pon tu API Key aquí directamente solo para esta prueba
API_KEY = "1abc53997c1b26e3b447796665e36e44" 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

def buscar_id_leagues_cup():
    url = f"{BASE_URL}/leagues"
    # Buscamos específicamente por el nombre
    querystring = {"search": "Leagues Cup"}
    
    response = requests.get(url, headers=HEADERS, params=querystring)
    datos = response.json().get("response", [])
    
    print(f"Se encontraron {len(datos)} ligas con ese nombre:\n")
    
    for liga in datos:
        id_liga = liga["league"]["id"]
        nombre = liga["league"]["name"]
        pais = liga["country"]["name"]
        
        print(f"👉 ID: {id_liga} | Torneo: {nombre} | País/Región: {pais}")

if __name__ == "__main__":
    buscar_id_leagues_cup()
