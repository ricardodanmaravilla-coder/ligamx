# Proveedores gratuitos de datos y cuotas — Liga MX

## Integracion

* **API-Football** (principal): `API_SPORTS_KEY`. Cuotas y fixtures; cuota gratuita condicionada a cobertura por temporada.
* **StrikerAPI** (respaldo automatico): `STRIKER_API_KEY`. Registro gratuito con 500 creditos/mes. El adaptador usa `list_games` (2 creditos) y `get_odds` (5 creditos) solo al necesitar respaldo. La fecha y ambos equipos deben coincidir; no reutiliza los IDs de API-Football.
* **Football-Data** (historico CSV): `from modules.external_free_sources import descargar_historico_football_data`. CSV gratuito de Mexico en https://www.football-data.co.uk/new/MEX.csv. Solo para historico y backtesting, NO para cuotas prepartido en vivo.
* **FootyStats**: https://footystats.org/es/mexico/liga-mx/xg — visualizacion parcial gratuita de xG, sin API gratuita garantizada.
* **FotMob**: https://www.fotmob.com/ — consulta manual de estadisticas, sin API publica gratuita garantizada.

## Configuracion

En **Google Cloud Run**, crear tres secretos en Secret Manager y montarlos como variables de entorno (no subir claves a GitHub):

```text
STRIKER_API_KEY_1 -> secreto striker-api-key-1:latest
STRIKER_API_KEY_2 -> secreto striker-api-key-2:latest
STRIKER_API_KEY_3 -> secreto striker-api-key-3:latest
API_SPORTS_KEY    -> secreto existente
```

Asignar al service account de Cloud Run acceso `roles/secretmanager.secretAccessor` sobre cada secreto. Configurar solo en la revisión del servicio autorizado. La variable antigua `STRIKER_API_KEY` sirve como compatibilidad si no están disponibles las tres nuevas.

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

## Uso de tres cuentas autorizadas

El usuario confirma contar con autorización de StrikerAPI para utilizar tres cuentas en este proyecto. El adaptador asigna cada partido de forma **determinista** a una de las claves `STRIKER_API_KEY_1`, `STRIKER_API_KEY_2`, `STRIKER_API_KEY_3` y utiliza esa misma cuenta para `list_games` y `get_odds`. No es un contador global ni un balanceador según créditos restantes; por seguridad NO reintenta con otra cuenta cuando hay errores 402/403/429.

**Pendiente antes de uso intensivo:** implementar presupuesto persistente por cuenta y mes (p. ej. Firestore + transacciones) y caché compartida entre instancias (p. ej. Firestore/Redis). Cloud Run puede ejecutar múltiples instancias y el almacenamiento en memoria no ofrece contabilización fiable. Los 500 créditos por cuenta son presupuestos independientes solo en la medida en que StrikerAPI lo haya aprobado expresamente; confirmar alcance de autorización y los cargos por herramienta. Nunca guardar claves ni la carta de autorización en el repositorio público.

**No verificado:** desplegar nueva revisión, configurar secretos, validar contrato de respuestas en vivo, revisar logs y ejecutar pruebas sobre un fixture real. Hasta entonces, el estado es código preparado pero no operativo en Cloud Run.
