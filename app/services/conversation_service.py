"""
Memoria de conversacion por numero de telefono.

En el notebook, Llama Stack llevaba esto por dentro: `client.conversations`
guardaba el historial del lado del servidor y solo haciamos referencia a un
`conversation_id` (celda 14). Al quitar Llama Stack, ese historial hay que
guardarlo nosotros -- aqui vive en la columna `historial` (JSONB) de la
tabla `conversaciones`, como una lista de mensajes {role, content} al
estilo de la API de Groq/OpenAI.

Se guarda solo texto (preguntas y respuestas finales), no los turnos
intermedios de function calling -- son necesarios dentro de una sola
llamada a `generar_respuesta` (ver llm_service.py) pero no aportan nada al
contexto de turnos futuros, y guardarlos infla el historial sin necesidad.
"""

import json

from app.database import consultar, ejecutar

MAX_TURNOS_HISTORIAL = 12  # ~6 intercambios usuario/asistente; suficiente contexto sin inflar cada llamada


def obtener_historial(numero: str) -> list[dict]:
    filas = consultar("SELECT historial FROM conversaciones WHERE numero = %s;", (numero,))
    if not filas:
        return []
    return filas[0][0]  # psycopg2 ya decodifica JSONB a list/dict de Python


def guardar_turno(numero: str, mensaje_usuario: str, respuesta_asistente: str) -> None:
    historial = obtener_historial(numero)
    historial.append({"role": "user", "content": mensaje_usuario})
    historial.append({"role": "assistant", "content": respuesta_asistente})
    historial = historial[-MAX_TURNOS_HISTORIAL:]

    ejecutar(
        """
        INSERT INTO conversaciones (numero, historial, actualizado_en)
        VALUES (%s, %s::jsonb, now())
        ON CONFLICT (numero) DO UPDATE SET historial = EXCLUDED.historial, actualizado_en = now();
        """,
        (numero, json.dumps(historial)),
    )
