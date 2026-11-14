"""
Conexion a Postgres (Supabase) y creacion del esquema.

En el notebook original, el estado (conversaciones, ids ya procesados,
catalogo de obligaciones, metricas, errores, bloqueos) vivia en listas y
diccionarios de Python en memoria -- se perdia cada vez que se reiniciaba
el runtime de Colab. Aqui ese mismo estado vive en tablas de Postgres, para
que sobreviva un redeploy o un reinicio del proceso en Railway.

`documentos_fiscales` (RAG) no se crea aqui: ya existe desde la migracion a
pgvector (ver scripts/migrar_base_fiscal.py), que es quien la crea.
"""

import logging

import psycopg2
import psycopg2.extensions

from app.config import SUPABASE_DB_URL

logger = logging.getLogger("fiscal_assistant.database")

_conexion: psycopg2.extensions.connection | None = None


def obtener_conexion() -> psycopg2.extensions.connection:
    """
    Regresa una conexion viva, reconectando si la anterior se cayo por
    inactividad -- mismo patron defensivo que `buscar_en_base_fiscal` ya
    usaba en el notebook (celda 18g) para su propia conexion.
    """
    global _conexion
    if _conexion is None or _conexion.closed:
        _conexion = psycopg2.connect(SUPABASE_DB_URL)
        _conexion.autocommit = True
        logger.info("Conexion a Postgres (re)establecida.")
    return _conexion


def ejecutar(query: str, parametros: tuple = ()) -> None:
    """Ejecuta una sentencia sin resultado esperado (INSERT/UPDATE/DDL), con un reintento si la conexion se cayo."""
    try:
        with obtener_conexion().cursor() as cur:
            cur.execute(query, parametros)
    except psycopg2.OperationalError:
        global _conexion
        _conexion = None
        with obtener_conexion().cursor() as cur:
            cur.execute(query, parametros)


def consultar(query: str, parametros: tuple = ()) -> list[tuple]:
    """Ejecuta un SELECT y regresa todas las filas, con el mismo reintento."""
    try:
        with obtener_conexion().cursor() as cur:
            cur.execute(query, parametros)
            return cur.fetchall()
    except psycopg2.OperationalError:
        global _conexion
        _conexion = None
        with obtener_conexion().cursor() as cur:
            cur.execute(query, parametros)
            return cur.fetchall()


def inicializar_esquema() -> None:
    """
    Crea (si no existen) las tablas de estado que en el notebook eran
    estructuras en memoria. Se llama una vez al arrancar la app
    (ver app/main.py, evento de startup). Todas las sentencias son
    idempotentes (IF NOT EXISTS), igual que la creacion de
    `documentos_fiscales` en el notebook original.
    """
    ejecutar("""
        CREATE TABLE IF NOT EXISTS conversaciones (
            numero TEXT PRIMARY KEY,
            historial JSONB NOT NULL DEFAULT '[]'::jsonb,
            actualizado_en TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    ejecutar("""
        CREATE TABLE IF NOT EXISTS mensajes_procesados (
            wamid TEXT PRIMARY KEY,
            numero TEXT NOT NULL,
            procesado_en TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    ejecutar("""
        CREATE TABLE IF NOT EXISTS catalogo_obligaciones (
            rfc TEXT PRIMARY KEY,
            negocio TEXT NOT NULL,
            pendientes JSONB NOT NULL DEFAULT '[]'::jsonb,
            fecha_limite DATE,
            adeudo NUMERIC NOT NULL DEFAULT 0
        );
    """)

    # Tabla unica para todo lo que en el notebook eran listas separadas
    # (mensajes_recibidos, respuestas_enviadas, bloqueos, errores_procesamiento,
    # metricas): un registro de eventos con `tipo` para distinguirlos, en vez
    # de cinco tablas casi identicas.
    ejecutar("""
        CREATE TABLE IF NOT EXISTS eventos (
            id BIGSERIAL PRIMARY KEY,
            tipo TEXT NOT NULL,
            numero TEXT,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            creado_en TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    ejecutar("CREATE INDEX IF NOT EXISTS idx_eventos_tipo ON eventos (tipo);")

    logger.info("Esquema de base de datos verificado/creado.")
