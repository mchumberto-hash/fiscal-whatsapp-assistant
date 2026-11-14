"""
Registro de eventos (mensajes, respuestas, bloqueos, errores, metricas) y
resumen basico -- reemplaza a las listas en memoria del notebook
(`mensajes_recibidos`, `respuestas_enviadas`, `bloqueos`,
`errores_procesamiento`, `metricas`) con una sola tabla `eventos` en
Postgres (ver app/database.py), distinguida por la columna `tipo`.

Guardar esto en una tabla en vez de en listas de Python es justo el punto
que el notebook ya señalaba como pendiente (celda 46): un agente en
produccion necesita poder responder "cuantos mensajes, cuanto tardan, que
porcentaje falla" sin depender de que el proceso siga vivo con las listas
intactas en RAM.
"""

import json
import logging
from collections import Counter

from app.database import consultar, ejecutar

logger = logging.getLogger("fiscal_assistant.metrics")


def registrar_evento(tipo: str, numero: str | None, payload: dict) -> None:
    ejecutar(
        "INSERT INTO eventos (tipo, numero, payload) VALUES (%s, %s, %s::jsonb);",
        (tipo, numero, json.dumps(payload, default=str)),
    )


def registrar_bloqueo(numero: str, texto: str, capa: str, detalle: str) -> None:
    registrar_evento("bloqueo_seguridad", numero, {"texto": texto, "capa": capa, "detalle": detalle})


def registrar_error(numero: str, texto: str, causa: str, detalle: str, error: Exception) -> None:
    registrar_evento("error_procesamiento", numero, {"texto": texto, "causa": causa, "detalle": detalle, "error": repr(error)})
    logger.error("Error procesando mensaje de %s (causa=%s): %s", numero, causa, detalle)


def registrar_metrica(numero: str, ruta: str, duracion_s: float) -> None:
    registrar_evento("metrica_mensaje", numero, {"ruta": ruta, "duracion_s": duracion_s})


def resumen_metricas(ultimos_n: int = 500) -> dict:
    """Mismo resumen que `resumen_metricas()` del notebook (celda 47), ahora leido desde la tabla en vez de una lista en RAM."""
    filas = consultar(
        "SELECT payload FROM eventos WHERE tipo = 'metrica_mensaje' ORDER BY id DESC LIMIT %s;",
        (ultimos_n,),
    )
    metricas = [fila[0] for fila in filas]
    total = len(metricas)
    if total == 0:
        return {"total": 0, "mensaje": "Aun no hay mensajes procesados."}

    por_ruta = Counter(m["ruta"] for m in metricas)
    errores = por_ruta.get("error", 0)
    tiempos_llama = [m["duracion_s"] for m in metricas if m["ruta"] == "llama"]

    filas_causas = consultar(
        "SELECT payload->>'causa' FROM eventos WHERE tipo = 'error_procesamiento' ORDER BY id DESC LIMIT %s;",
        (ultimos_n,),
    )
    causas = Counter(fila[0] for fila in filas_causas)

    return {
        "total": total,
        "volumen_por_ruta": dict(por_ruta),
        "tasa_error": round(errores / total, 4),
        "tiempo_promedio_s": round(sum(m["duracion_s"] for m in metricas) / total, 2),
        "tiempo_promedio_llama_s": round(sum(tiempos_llama) / len(tiempos_llama), 2) if tiempos_llama else None,
        "causas_de_error": dict(causas),
    }
