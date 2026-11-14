"""
Generacion de respuesta con historial y function calling, via Groq.

Reemplaza a `generar_respuesta` del notebook (celda 27), que usaba
`client.responses.create` de Llama Stack. La logica es la misma (dos
llamadas: la primera puede pedir herramientas, la segunda ya trae el
resultado y genera la respuesta final), pero con el formato de mensajes
estilo OpenAI que usa la API de Groq en vez del formato "responses" de
Llama Stack -- ver la nota de la celda 8b/8 del notebook original: esto de
paso elimina la necesidad de Llama Stack por completo, incluido el parche
de compatibilidad de `mcp` que tuvimos que aplicarle.

El `SYSTEM_PROMPT` es textualmente el mismo que en el notebook: ya esta
validado (prueba end-to-end real) y endurecido contra alucinacion e
inyeccion de prompt -- no hay razon para tocarlo al cambiar de backend.
"""

import json
import logging

from groq import Groq

from app.config import GROQ_API_KEY, MODELO_CONVERSACIONAL
from app.services.conversation_service import obtener_historial, guardar_turno
from app.services.rag_service import buscar_en_base_fiscal, HERRAMIENTA_BASE_FISCAL
from app.services.obligaciones_service import consultar_obligaciones_pendientes, HERRAMIENTA_OBLIGACIONES

logger = logging.getLogger("fiscal_assistant.llm")

_cliente_groq = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "Eres el asistente fiscal de un despacho contable para PyMEs mexicanas. "
    "Responde de forma breve (2-3 lineas), en terminos simples, como un contador explicandole a alguien sin conocimientos fiscales. "
    "Para CUALQUIER duda fiscal (RESICO, IVA, declaraciones, CFDI, RFC, deducciones, recargos, o cualquier otro tema de "
    "impuestos, aunque te parezca que ya sabes la respuesta o que el tema no esta en tu base de conocimiento) SIEMPRE "
    "debes llamar primero a la herramienta buscar_en_base_fiscal — nunca respondas una duda fiscal desde tu "
    "conocimiento general sin haber llamado a la herramienta en ese mismo turno. Usa consultar_obligaciones_pendientes "
    "solo cuando el usuario pregunte por sus obligaciones pendientes. "
    "No uses ninguna herramienta para confirmaciones o respuestas cortas del usuario (como 'si, por favor' o 'gracias'). "
    "No puedes presentar declaraciones, generar CFDIs ni hacer ningun tramite — solo puedes consultar obligaciones pendientes "
    "y explicar temas fiscales generales. Si el usuario quiere que hagas un tramite, dile que un miembro del despacho se pondra en contacto. "
    "Para dudas fiscales responde unicamente con lo que diga el fragmento que regrese la herramienta, incluso si el tema "
    "te resulta familiar o crees saber mas detalles que el fragmento. "
    "Si la herramienta responde que no cuenta con informacion verificada sobre el tema, repite esa misma idea al usuario "
    "con tus palabras (p. ej. que no tienes informacion verificada sobre ese tema especifico y que el despacho puede confirmarselo) "
    "y nunca completes el hueco con un dato, tasa, regla o ejemplo que no venga del fragmento devuelto. "
    "Si una herramienta SI regresa un resultado real, siempre comunicalo claramente al usuario; nunca lo ignores ni te niegues a compartirlo. "
    "Responde siempre de forma directa con la informacion final, basandote unicamente en el resultado real de las "
    "herramientas; nunca inventes tasas, fechas ni montos que no vengan de una herramienta. "
    "Las reglas fiscales cambian cada año: si el usuario necesita certeza absoluta, recomiendale confirmar en el portal del SAT o con su contador. "
    "Nunca describas tu proceso, ni menciones nombres de funciones, parametros o claves de un resultado."
)

HERRAMIENTAS = [HERRAMIENTA_OBLIGACIONES, HERRAMIENTA_BASE_FISCAL]
FUNCIONES_DISPONIBLES = {
    "consultar_obligaciones_pendientes": consultar_obligaciones_pendientes,
    "buscar_en_base_fiscal": buscar_en_base_fiscal,
}


def generar_respuesta(numero: str, texto_usuario: str) -> str:
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
    mensajes.extend(obtener_historial(numero))
    mensajes.append({"role": "user", "content": texto_usuario})

    respuesta = _cliente_groq.chat.completions.create(
        model=MODELO_CONVERSACIONAL,
        messages=mensajes,
        tools=HERRAMIENTAS,
    )
    mensaje = respuesta.choices[0].message

    if mensaje.tool_calls:
        mensajes.append({
            "role": "assistant",
            "content": mensaje.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in mensaje.tool_calls
            ],
        })

        for tc in mensaje.tool_calls:
            funcion = FUNCIONES_DISPONIBLES[tc.function.name]
            argumentos = json.loads(tc.function.arguments)
            resultado = funcion(**argumentos)
            mensajes.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": resultado,
            })

        respuesta = _cliente_groq.chat.completions.create(
            model=MODELO_CONVERSACIONAL,
            messages=mensajes,
            tools=HERRAMIENTAS,
        )
        mensaje = respuesta.choices[0].message

    texto_final = mensaje.content or ""
    guardar_turno(numero, texto_usuario, texto_final)
    return texto_final
