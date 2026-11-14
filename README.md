# Asistente Fiscal por WhatsApp para PyMEs Mexicanas

Agente conversacional por WhatsApp para despachos contables: responde dudas
fiscales con una base de conocimiento verificada (RAG sobre pgvector),
consulta en tiempo real si un cliente tiene declaraciones pendientes
(function calling sobre un catálogo real), y aplica dos capas de seguridad
de contenido (Llama Guard + Prompt Guard) antes de contestar.

Este repositorio es la versión de producción del prototipo que se
construyó y validó en un notebook de Google Colab (evaluación formal de
retrieval, migración a pgvector, prueba end-to-end real por WhatsApp). El
caso de estudio completo, con las métricas de esa validación, está en
`docs/caso_estudio_es.pdf` (y su versión en inglés).

## Qué cambió al salir del notebook

| | Prototipo (Colab) | Este repo |
|---|---|---|
| Modelo conversacional | Ollama local (`llama3.1:8b`) | API de Groq (mismo modelo, hospedado) |
| Llama Guard | Ollama local (`llama-guard3`) | API de Groq (`llama-guard-3-8b`) |
| Orquestación / function calling | Llama Stack | Llamadas directas a la API de Groq (formato OpenAI) |
| Memoria de conversación | `client.conversations` de Llama Stack | Tabla `conversaciones` en Postgres |
| Catálogo de obligaciones | Diccionario fijo en memoria | Tabla `catalogo_obligaciones` en Postgres |
| Deduplicación de mensajes (`wamid`) | `set()` en memoria | Tabla `mensajes_procesados` (constraint UNIQUE) |
| Métricas / errores / bloqueos | Listas en memoria | Tabla `eventos` en Postgres |
| URL pública | Túnel de ngrok | URL fija con HTTPS (Railway) |
| Secretos | Colab Secrets | Variables de entorno |

Se eliminó Llama Stack por completo, lo cual también elimina el parche de
compatibilidad `McpError`/`streamablehttp_client` que hubo que aplicarle
en el notebook (era un workaround de una incompatibilidad entre
llama-stack 0.7.3 y el SDK de `mcp` — al quitar Llama Stack, deja de
aplicar).

La base de conocimiento fiscal (RAG) y su migración a pgvector ya estaban
en su forma de producción antes de salir de Colab, así que esa parte casi
no cambió — ver `app/services/rag_service.py`.

## Arquitectura

```
WhatsApp Business API → webhook (FastAPI) → Llama Guard (entrada, Groq)
                                                   ↓
                                    reglas fijas / Groq + herramientas
                                      ├─ buscar_en_base_fiscal (pgvector)
                                      └─ consultar_obligaciones_pendientes (Postgres)
                                                   ↓
                                      Llama Guard (salida, Groq) → WhatsApp
```

## Estructura del proyecto

```
app/
├── main.py                      # FastAPI: endpoints del webhook + /metricas
├── config.py                    # variables de entorno
├── database.py                  # conexión a Postgres + creación del esquema
├── orchestrator.py               # flujo de decisión (seguridad → reglas → LLM → seguridad)
└── services/
    ├── rag_service.py            # búsqueda en la base fiscal (pgvector)
    ├── obligaciones_service.py   # consulta de obligaciones pendientes
    ├── safety_service.py         # Llama Guard + Prompt Guard
    ├── conversation_service.py   # memoria de conversación por número
    ├── llm_service.py            # generación de respuesta + function calling (Groq)
    ├── whatsapp_service.py       # envío de mensajes
    └── metrics_service.py        # eventos, métricas, resumen
scripts/
├── migrar_base_fiscal.py         # genera embeddings y los sube a pgvector
├── evaluar_umbral.py             # (dev) recalibra el umbral de similitud
└── seed_catalogo_demo.py         # siembra 3 RFCs de ejemplo para pruebas
```

## Configuración local

1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env` y llena las variables (ver comentarios en el archivo)
4. Migra la base fiscal: `python -m scripts.migrar_base_fiscal`
5. Siembra el catálogo de ejemplo: `python -m scripts.seed_catalogo_demo`
6. Levanta el servidor: `uvicorn app.main:app --reload`

## Despliegue en Railway

1. Conecta este repo a un proyecto nuevo de Railway.
2. En Settings → Variables, agrega las mismas variables de `.env.example`.
3. Railway detecta el `Procfile` y expone una URL fija con HTTPS — esa URL
   + `/webhook` es el Callback URL que se configura en Meta for Developers
   (WhatsApp → Configuración de la API → Configurar webhooks).
4. Antes del primer despliegue, corre `migrar_base_fiscal.py` y
   `seed_catalogo_demo.py` apuntando a la misma `SUPABASE_DB_URL` (desde tu
   máquina local, o como un one-off en Railway).

## Limitaciones conocidas

- **Locks por número** (`app/orchestrator.py`) son un diccionario en
  memoria del proceso: correcto con una sola instancia de Railway, pero no
  coordina entre instancias si más adelante se escala horizontalmente.
- **Un número de WhatsApp por despacho**: con varios despachos como
  clientes, cada uno necesita su propio WABA verificado — este repo no
  implementa enrutamiento multi-tenant sobre un único número.
- **Contenido fiscal ilustrativo**: los documentos de `scripts/migrar_base_fiscal.py`
  son para el ejercicio 2026, citados por fuente, pero no son el texto
  oficial del SAT/DOF — un contador certificado debe validarlos antes de
  usarlos con un cliente real.
