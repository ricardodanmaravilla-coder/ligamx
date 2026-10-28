# Proveedores gratuitos de datos y cuotas — Liga MX

## Integracion

* **API-Football** (principal): `API_SPORTS_KEY`. Cuotas y fixtures; cuota gratuita condicionada a cobertura por temporada.
* **StrikerAPI** (respaldo automatico): `STRIKER_API_KEY`. Registro gratuito con 500 creditos/mes. El adaptador usa `list_games` (2 creditos) y `get_odds` (5 creditos) solo al necesitar respaldo. La fecha y ambos equipos deben coincidir; no reutiliza los IDs de API-Football.
* **Football-Data** (historico CSV): `from modules.external_free_sources import descargar_historico_football_data`. CSV gratuito de Mexico en https://www.football-data.co.uk/new/MEX.csv. Solo para historico y backtesting, NO para cuotas prepartido en vivo.
* **FootyStats**: https://footystats.org/es/mexico/liga-mx/xg — visualizacion parcial gratuita de xG, sin API gratuita garantizada.
* **FotMob**: https://www.fotmob.com/ — consulta manual de estadisticas, sin API publica gratuita garantizada.

## Configuracion

En Streamlit Cloud > App settings > Secrets, agregar claves sin subirlas al repositorio:

```toml
API_SPORTS_KEY = "TU_CLAVE_API_FOOTBALL"
STRIKER_API_KEY = "TU_CLAVE_STRIKER"
```

## Uso del CSV historico sin sobrescribir la base existente

```python
from modules.external_free_sources import descargar_historico_football_data
df = descargar_historico_football_data()
df.to_csv("data/football_data_mexico_raw.csv", index=False)
```

Conservar la tabla original de features `data/historico_ligamx_completo.csv`; los encabezados, equipos y mercados del CSV externo deben auditarse y normalizarse antes del merge. No usar odds historicas para declarar oportunidades actuales.

## Seguridad y calidad

* Nunca publicar API keys ni incluirlas en logs.
* El respaldo StrikerAPI solo sirve si un partido tiene exactamente una coincidencia por fecha, local y visitante.
* Se elige una sola casa para el snapshot 1X2; no se mezclan cuotas.
* Sin precios completos, sin datos o sin coincidencia fiable => **NO BET**.
* El modulo NO infiere lineas de corners o tarjetas cuando el proveedor no las publica.
* Solo son accionables snapshots prepartido de como maximo 3 horas, con estado scheduled. Las cuotas no garantizan vigencia: revisar `captured_at` del snapshot antes de publicar picks reales.
* Los endpoints consumen creditos; el resultado puede variar con la cobertura gratuita y el presupuesto.

## Referencias

- https://strikerapi.com/liga-mx
- https://strikerapi.com/docs/list-games
- https://strikerapi.com/docs/get-odds
- https://www.football-data.co.uk/mexico.php
- https://www.api-football.com/pricing

## Estado

Adaptador StrikerAPI agregado, fallback en `modules/odds_engine.py` conectado a `app.py` y `app_v2.py`. La integracion no se ha validado con credenciales reales y no hay pruebas de despliegue. Football-Data tiene descargador separado, sin alterar los datos historicos originales.

## Uso de varias cuentas (Terminos vigentes 18 octubre 2026)

StrikerAPI prohibe expresamente eludir limites de creditos, rate limits o cuotas usando muchas cuentas. No implementar rotacion de claves de cuentas diferentes para ampliar artificialmente el plan gratuito. Mantener una unica cuenta autorizada para este proyecto; las otras claves no se integran ni se cargan en Cloud Run sin permiso escrito del proveedor. Contactar support@strikerapi.com si existe una necesidad legitima de varias cuentas. Ver https://strikerapi.com/terms (seccion 7). La clave activa se configura en Secret Manager como STRIKER_API_KEY.
