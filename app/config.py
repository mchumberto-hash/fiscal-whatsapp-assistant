"""
Configuracion centralizada, leida de variables de entorno.

Reemplaza a `userdata.get(...)` de Colab Secrets (ver notebook original,
celda 2): en produccion las credenciales viven en variables de entorno
(en Railway, se configuran en Settings > Variables del servicio), nunca
en el codigo ni en un archivo que se suba al repo.
"""

import os

from dotenv import load_dotenv

# Solo tiene efecto en desarrollo local: carga las variables del archivo
# .env al entorno del proceso (python-dotenv nunca sobreescribe una
# variable que ya exista en el entorno). En Railway no hay archivo .env
# -- las variables ya estan inyectadas directo al proceso -- asi que esta
# llamada simplemente no encuentra nada que cargar y no hace nada.
load_dotenv()


class ConfigError(RuntimeError):
    """Falta una variable de entorno requerida."""


def _requerida(nombre: str) -> str:
    valor = os.environ.get(nombre)
    if not valor:
        raise ConfigError(
            f"Falta la variable de entorno {nombre}. Revisa .env.example "
            f"y configura las variables en tu plataforma de despliegue."
        )
    return valor


def _opcional(nombre: str, default: str | None = None) -> str | None:
    return os.environ.get(nombre, default)


# --- WhatsApp / Meta ---
VERIFY_TOKEN = _requerida("VERIFY_TOKEN")
WHATSAPP_TOKEN = _requerida("WHATSAPP_ACCESS_TOKEN")
PHONE_ID = _requerida("PHONE_NUMBER_ID")
WABA_ID = _opcional("WABA_TOKEN_ID")

# --- Groq (modelo conversacional + Llama Guard) ---
GROQ_API_KEY = _requerida("GROQ_API_KEY")
MODELO_CONVERSACIONAL = _opcional("MODELO_CONVERSACIONAL", "llama-3.1-8b-instant")
MODELO_LLAMA_GUARD = _opcional("MODELO_LLAMA_GUARD", "llama-guard-3-8b")

# --- Supabase / Postgres (RAG + estado) ---
SUPABASE_DB_URL = _requerida("SUPABASE_DB_URL")

# --- Prompt Guard (opcional; si falta, el agente sigue solo con Llama Guard) ---
HF_TOKEN = _opcional("HF_TOKEN")
PROMPT_GUARD_MODELO = _opcional("PROMPT_GUARD_MODELO", "meta-llama/Llama-Prompt-Guard-2-86M")

# --- Umbral de similitud para el RAG fiscal ---
# Calibrado en el notebook original (celda 18d) con el set de evaluacion de
# recall@k / tasa de falso positivo: 0.56 -> 80% recall, 0% falsos positivos.
# Si se agregan o editan documentos en documentos_fiscales, hay que volver a
# correr scripts/evaluar_umbral.py y actualizar este valor (o la variable de
# entorno UMBRAL_BASE_FISCAL) con el resultado nuevo.
UMBRAL_BASE_FISCAL = float(_opcional("UMBRAL_BASE_FISCAL", "0.56"))
