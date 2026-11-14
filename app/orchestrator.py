"""
Flujo de decision y procesamiento de cada mensaje entrante.

Equivale a las celdas 32-34 del notebook (`obtener_lock`,
`decidir_y_responder`, `procesar_y_responder`): mismo orden siempre
-- seguridad de entrada, reglas fijas, Llama con herramientas, seguridad
de salida -- y la misma clasificacion de errores, adaptada a las
excepciones del SDK de Groq en vez de las de `llama_stack_client`.

LIMITACION CONOCIDA (igual que en el notebook, pero ahora con una
implicacion distinta): los locks por numero (`_locks_por_numero`) siguen
siendo un diccionario en memoria del proceso. Mientras Railway corra una
sola instancia esto serializa correctamente los mensajes de un mismo
numero. Si mas adelante se escala a varias instancias, un lock en memoria
de cada proceso ya NO coordina entre instancias -- en ese punto hay que
pasar a un lock distribuido (p. ej. un advisory lock de Postgres) en vez
de `threading.Lock()`.
"""

import logging
import threading
import time

import groq
import requests

from app.services.llm_service import generar_respuesta
from app.services.metrics_service import registrar_bloqueo, registrar_error, registrar_metrica
from app.services.safety_service import (
    MENSAJE_BLOQUEO,
    MENSAJE_RESPUESTA_BLOQUEADA,
    es_inyeccion,
    revisar_con_llama_guard,
)
from app.services.whatsapp_service import enviar_respuesta

logger = logging.getLogger("fiscal_assistant.orchestrator")

FRASES_ESCALAMIENTO = ["hablar con alguien", "hablar con una persona", "agente humano", "asistente humano"]
FRASES_SALUDO = ["hola", "buenas", "buenos dias", "buenos días", "buenas tardes", "buenas noches", "que tal", "buen dia", "buen día"]

_locks_por_numero: dict[str, threading.Lock] = {}


def _obtener_lock(numero: str) -> threading.Lock:
    if numero not in _locks_por_numero:
        _locks_por_numero[numero] = threading.Lock()
    return _locks_por_numero[numero]


def _es_solo_saludo(texto_low: str) -> bool:
    return texto_low.strip(" ,.!¡¿?") in FRASES_SALUDO


def _enviar_aviso_de_error(numero: str) -> None:
    try:
        enviar_respuesta(numero, "Estamos teniendo un problema tecnico. Intenta de nuevo en un momento.")
    except Exception:
        pass  # si ni el aviso de error se pudo mandar, no hay mas que intentar aqui


def decidir_y_responder(numero: str, texto: str) -> str:
    texto_low = texto.lower()

    # 1. Seguridad sobre la entrada
    if es_inyeccion(texto):
        registrar_bloqueo(numero, texto, "prompt_guard", "inyeccion de prompt")
        enviar_respuesta(numero, MENSAJE_BLOQUEO)
        return "bloqueado_prompt_guard"

    seguro, categoria = revisar_con_llama_guard([{"role": "user", "content": texto}])
    if not seguro:
        registrar_bloqueo(numero, texto, "llama_guard_entrada", categoria)
        enviar_respuesta(numero, MENSAJE_BLOQUEO)
        return "bloqueado_llama_guard_entrada"

    # 2. Reglas fijas
    if any(frase in texto_low for frase in FRASES_ESCALAMIENTO):
        enviar_respuesta(numero, "Claro, en un momento un miembro del despacho te contacta directamente.")
        return "regla_escalamiento"
    if _es_solo_saludo(texto_low):
        enviar_respuesta(numero, "Hola, soy el asistente fiscal del despacho. ¿En que puedo ayudarte?")
        return "regla_saludo"

    # 3. Llama (Groq) con herramientas
    respuesta = generar_respuesta(numero, texto)

    # 4. Seguridad sobre la salida
    seguro, categoria = revisar_con_llama_guard([
        {"role": "user", "content": texto},
        {"role": "assistant", "content": respuesta},
    ])
    if not seguro:
        registrar_bloqueo(numero, respuesta, "llama_guard_salida", categoria)
        enviar_respuesta(numero, MENSAJE_RESPUESTA_BLOQUEADA)
        return "bloqueado_llama_guard_salida"

    enviar_respuesta(numero, respuesta)
    return "llama"


def procesar_y_responder(numero: str, texto: str) -> None:
    with _obtener_lock(numero):
        inicio = time.perf_counter()
        ruta = "error"
        try:
            ruta = decidir_y_responder(numero, texto)

        except groq.APITimeoutError as e:
            registrar_error(numero, texto, "llm_timeout", "Groq conecto pero la generacion tardo demasiado.", e)
            _enviar_aviso_de_error(numero)

        except groq.RateLimitError as e:
            registrar_error(numero, texto, "llm_rate_limit", "Se alcanzo el limite de tasa de la API de Groq.", e)
            _enviar_aviso_de_error(numero)

        except groq.APIConnectionError as e:
            registrar_error(numero, texto, "llm_conexion", "No se pudo conectar a la API de Groq.", e)
            _enviar_aviso_de_error(numero)

        except groq.APIStatusError as e:
            registrar_error(numero, texto, "llm_error_http", f"Groq respondio con error HTTP {e.status_code}.", e)
            _enviar_aviso_de_error(numero)

        except KeyError as e:
            registrar_error(numero, texto, "herramienta_no_reconocida", f"El modelo pidio ejecutar una herramienta que no existe: {e}.", e)
            _enviar_aviso_de_error(numero)

        except requests.exceptions.RequestException as e:
            # No se intenta avisar al usuario: si fallo mandar el mensaje a WhatsApp, el aviso fallaria igual
            registrar_error(numero, texto, "whatsapp_api_error", "Fallo el envio a la API de WhatsApp (graph.facebook.com).", e)

        except Exception as e:
            registrar_error(numero, texto, "desconocido", "Error no clasificado.", e)
            _enviar_aviso_de_error(numero)

        finally:
            registrar_metrica(numero, ruta, round(time.perf_counter() - inicio, 2))
