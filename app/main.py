"""
App de FastAPI: los mismos dos endpoints del webhook que en el notebook
(celdas 29-30), mas un endpoint de metricas para el reporte que se le
puede ensenar a un despacho (ver app/services/metrics_service.py).

Lo que cambia respecto al notebook:
- No hay ngrok: Railway asigna una URL fija con HTTPS directamente.
- No hay `ids_procesados` en memoria: la deduplicacion de wamids repetidos
  (Meta reenvia el mismo mensaje si no se responde rapido) usa la tabla
  `mensajes_procesados`, vía una constraint UNIQUE -- sobrevive un redeploy.
- El esquema de base de datos se crea/verifica una vez al arrancar
  (evento `startup`), no celda por celda.
"""

import hashlib
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import PlainTextResponse
import psycopg2

from app.config import METRICAS_TOKEN, VERIFY_TOKEN, WHATSAPP_APP_SECRET
from app.database import ejecutar, inicializar_esquema
from app.orchestrator import procesar_y_responder
from app.services.metrics_service import registrar_evento, resumen_metricas
from app.services.whatsapp_service import normalizar_numero_mx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fiscal_assistant.main")


def _firma_valida(cuerpo_crudo: bytes, firma_header: str | None) -> bool:
    """Valida que un POST a /webhook realmente venga de Meta, comparando el
    HMAC-SHA256 del cuerpo crudo (con el App Secret como llave) contra el
    header X-Hub-Signature-256 que Meta manda en cada entrega.

    Sin esto, cualquiera que conozca la URL del webhook podria mandar
    mensajes falsos y hacer que el asistente responda (usando el token
    permanente de WhatsApp) a numeros arbitrarios.
    """
    if not WHATSAPP_APP_SECRET:
        logger.warning(
            "WHATSAPP_APP_SECRET no configurado: no se esta validando la "
            "firma de Meta en POST /webhook."
        )
        return True
    if not firma_header or not firma_header.startswith("sha256="):
        return False
    firma_esperada = hmac.new(
        WHATSAPP_APP_SECRET.encode(), cuerpo_crudo, hashlib.sha256
    ).hexdigest()
    firma_recibida = firma_header[len("sha256="):]
    return hmac.compare_digest(firma_esperada, firma_recibida)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    inicializar_esquema()
    logger.info("Asistente fiscal listo.")
    yield


app = FastAPI(title="Asistente Fiscal WhatsApp", lifespan=_lifespan)


@app.get("/webhook")
async def verificar_webhook(request: Request):
    params = request.query_params
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(params.get("hub.challenge"))
    return {"error": "token invalido"}


@app.post("/webhook")
async def recibir_mensaje(request: Request, background_tasks: BackgroundTasks):
    cuerpo_crudo = await request.body()
    if not _firma_valida(cuerpo_crudo, request.headers.get("x-hub-signature-256")):
        logger.warning("Firma de Meta invalida en POST /webhook; peticion rechazada.")
        return PlainTextResponse("firma invalida", status_code=403)

    body = await request.json()
    try:
        mensaje = body["entry"][0]["changes"][0]["value"]["messages"][0]
        wamid = mensaje["id"]
        numero = normalizar_numero_mx(mensaje["from"])
        texto = mensaje["text"]["body"]

        if not _marcar_procesado(wamid, numero):
            return {"status": "ok"}  # Meta reenvio el mismo mensaje; ya se proceso, se ignora

        registrar_evento("mensaje_recibido", numero, {"wamid": wamid, "texto": texto})
        background_tasks.add_task(procesar_y_responder, numero, texto)
    except (KeyError, IndexError):
        pass
    return {"status": "ok"}


def _marcar_procesado(wamid: str, numero: str) -> bool:
    """Intenta registrar el wamid; regresa False si ya existia (mensaje duplicado reenviado por Meta)."""
    try:
        ejecutar("INSERT INTO mensajes_procesados (wamid, numero) VALUES (%s, %s);", (wamid, numero))
        return True
    except psycopg2.errors.UniqueViolation:
        return False


def _metricas_autorizadas(token_recibido: str | None) -> bool:
    """Protege /metricas con un token simple por query param. Opcional por
    compatibilidad: si METRICAS_TOKEN no esta configurado, deja pasar pero
    avisa en el log (mismo patron que _firma_valida arriba)."""
    if not METRICAS_TOKEN:
        logger.warning("METRICAS_TOKEN no configurado: /metricas sigue siendo publico.")
        return True
    return token_recibido is not None and hmac.compare_digest(token_recibido, METRICAS_TOKEN)


@app.get("/metricas")
async def ver_metricas(request: Request):
    """Reporte basico -- mismo contenido que `resumen_metricas()` del notebook (celda 47), ahora como endpoint.
    Protegido por METRICAS_TOKEN: /metricas?token=... (ver _metricas_autorizadas)."""
    if not _metricas_autorizadas(request.query_params.get("token")):
        return PlainTextResponse("no autorizado", status_code=403)
    return resumen_metricas()


@app.get("/salud")
async def salud():
    return {"status": "ok"}
