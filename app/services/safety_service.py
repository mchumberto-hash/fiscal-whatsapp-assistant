"""
Capa de seguridad: Llama Guard (contenido) y Prompt Guard (inyeccion de prompt).

Llama Guard dejo de correr en Ollama local y ahora es una llamada mas a la
API de Groq (modelo `llama-guard-3-8b`, el mismo peso que se usaba vía
Ollama) -- el formato de salida del modelo ("safe" o "unsafe" + categoria
como "S2") es parte del propio modelo, no de quien lo sirve, asi que el
parseo es identico al del notebook (celda 22).

Prompt Guard sigue el mismo patron try/except del notebook (celda 24): si
no esta disponible (falta HF_TOKEN, o no se acepto la licencia del modelo
en Hugging Face), el agente avisa por log y continua solo con Llama Guard.
A diferencia del notebook, aqui SI vale la pena intentar activarlo en
produccion: Llama-Prompt-Guard-2-86M es un modelo de clasificacion de 86M
de parametros (no el modelo conversacional de 8B), corre rapido en CPU, y
Railway no necesita GPU para el -- la unica razon por la que seguia
desactivado en Colab era el acceso restringido en Hugging Face, no un
problema de rendimiento.
"""

import logging

from groq import Groq

from app.config import GROQ_API_KEY, MODELO_LLAMA_GUARD, PROMPT_GUARD_MODELO, HF_TOKEN

logger = logging.getLogger("fiscal_assistant.safety")

MENSAJE_BLOQUEO = "Lo siento, no puedo ayudar con ese mensaje. Si necesitas apoyo del despacho, escribeme tu duda."
MENSAJE_RESPUESTA_BLOQUEADA = "No puedo compartir esa respuesta. Un miembro del despacho se pondra en contacto contigo."

_cliente_groq = Groq(api_key=GROQ_API_KEY)


def revisar_con_llama_guard(mensajes: list[dict]) -> tuple[bool, str | None]:
    respuesta = _cliente_groq.chat.completions.create(model=MODELO_LLAMA_GUARD, messages=mensajes)
    veredicto = respuesta.choices[0].message.content.strip().lower()
    if veredicto.startswith("safe"):
        return True, None
    if veredicto.startswith("unsafe"):
        partes = veredicto.split()
        return False, partes[1].upper() if len(partes) > 1 else "sin_categoria"
    raise ValueError(f"Veredicto inesperado de Llama Guard: {veredicto!r}")


# --- Prompt Guard (opcional) ---

prompt_guard_activo = False
_pg_tokenizer = None
_pg_modelo = None

try:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN no configurado")

    from huggingface_hub import login
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    login(token=HF_TOKEN)
    _pg_tokenizer = AutoTokenizer.from_pretrained(PROMPT_GUARD_MODELO)
    _pg_modelo = AutoModelForSequenceClassification.from_pretrained(PROMPT_GUARD_MODELO)
    prompt_guard_activo = True
    logger.info("Prompt Guard listo. Etiquetas del modelo: %s", _pg_modelo.config.id2label)
except Exception as e:
    logger.warning("Prompt Guard no disponible, el agente continuara solo con Llama Guard: %r", e)


def es_inyeccion(texto: str) -> bool:
    if not prompt_guard_activo:
        return False
    import torch

    entradas = _pg_tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = _pg_modelo(**entradas).logits
    etiqueta = _pg_modelo.config.id2label[logits.argmax().item()].upper()
    return etiqueta not in ("BENIGN", "LABEL_0")
