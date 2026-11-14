"""
Busqueda en la base de conocimiento fiscal (RAG) contra pgvector.

Equivale a la version de produccion de `buscar_en_base_fiscal` del
notebook (celda 18g): la migracion a pgvector ya estaba hecha antes de
sacar el proyecto de Colab, asi que esta funcion cambia muy poco -- solo
usa `app.database` en vez de abrir su propia conexion a psycopg2, y el
formato de la herramienta para function calling pasa del formato plano de
Llama Stack al formato anidado que usa la API de Groq (OpenAI-compatible):
{"type": "function", "function": {...}} en vez de {"type": "function", ...}.

El modelo de embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) y el
umbral calibrado (0.56, ver app/config.py) son los mismos que en el
notebook -- la migracion de retrieval NO fue una recalibracion, y eso
sigue siendo cierto aqui.
"""

import logging

from sentence_transformers import SentenceTransformer

from app.config import UMBRAL_BASE_FISCAL
from app.database import consultar

logger = logging.getLogger("fiscal_assistant.rag")

MENSAJE_SIN_INFORMACION = (
    "No cuento con informacion verificada sobre ese tema; un miembro del "
    "despacho puede confirmartelo directamente."
)

# Cargar el modelo de embeddings una sola vez, al importar este modulo
# (igual que en el notebook, celda 16) -- no en cada llamada.
logger.info("Cargando modelo de embeddings (paraphrase-multilingual-MiniLM-L12-v2)...")
_modelo_embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
logger.info("Modelo de embeddings listo.")


def buscar_en_base_fiscal(pregunta: str, k: int = 2) -> str:
    embedding_pregunta = _modelo_embeddings.encode([pregunta], normalize_embeddings=True)[0]
    vector_str = "[" + ",".join(f"{x:.8f}" for x in embedding_pregunta) + "]"

    resultados = consultar(
        """
        SELECT contenido, 1 - (embedding <=> %s::vector) AS similitud
        FROM documentos_fiscales
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (vector_str, vector_str, k),
    )

    relevantes = [contenido for contenido, similitud in resultados if similitud >= UMBRAL_BASE_FISCAL]
    if not relevantes:
        return MENSAJE_SIN_INFORMACION
    return "Fragmentos de la base fiscal: " + " | ".join(relevantes)


# Formato de herramienta para la API de Groq (chat.completions.create con
# tools=[...]): a diferencia del formato plano que usaba Llama Stack, Groq
# (compatible con OpenAI) envuelve la definicion bajo la clave "function".
HERRAMIENTA_BASE_FISCAL = {
    "type": "function",
    "function": {
        "name": "buscar_en_base_fiscal",
        "description": (
            "Busca en la base de conocimiento fiscal la respuesta a dudas sobre "
            "RESICO, declaracion anual, pagos provisionales, IVA, deducciones "
            "personales, CFDI, recargos y multas, e inscripcion al RFC."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pregunta": {
                    "type": "string",
                    "description": "La pregunta del usuario sobre el tema fiscal, en texto libre.",
                }
            },
            "required": ["pregunta"],
        },
    },
}
