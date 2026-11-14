"""
Siembra la tabla `catalogo_obligaciones` con los 3 RFCs de ejemplo del
notebook original (celda 20) -- SOLO para pruebas, no son datos reales de
ningun despacho. Al conectar un despacho real, su catalogo real reemplaza
esto (idealmente sincronizado desde su propio sistema, no cargado a mano).

Correr con: python -m scripts.seed_catalogo_demo
"""

import json

import psycopg2

from app.config import SUPABASE_DB_URL

CATALOGO_DEMO = {
    "GARC800101AB1": {
        "negocio": "Papelería García",
        "pendientes": ["declaración mensual de septiembre 2026"],
        "fecha_limite": "2026-10-17",
        "adeudo": 0,
    },
    "LOPM900512XY2": {
        "negocio": "Consultoría López",
        "pendientes": [],
        "fecha_limite": None,
        "adeudo": 0,
    },
    "RAMJ850923TT9": {
        "negocio": "Taller Mecánico Ramírez",
        "pendientes": ["declaración mensual de agosto 2026", "declaración mensual de septiembre 2026"],
        "fecha_limite": "2026-10-17",
        "adeudo": 1250.00,
    },
}


def main() -> None:
    conexion = psycopg2.connect(SUPABASE_DB_URL)
    conexion.autocommit = True
    cursor = conexion.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS catalogo_obligaciones (
            rfc TEXT PRIMARY KEY,
            negocio TEXT NOT NULL,
            pendientes JSONB NOT NULL DEFAULT '[]'::jsonb,
            fecha_limite DATE,
            adeudo NUMERIC NOT NULL DEFAULT 0
        );
    """)

    for rfc, datos in CATALOGO_DEMO.items():
        cursor.execute(
            """
            INSERT INTO catalogo_obligaciones (rfc, negocio, pendientes, fecha_limite, adeudo)
            VALUES (%s, %s, %s::jsonb, %s, %s)
            ON CONFLICT (rfc) DO UPDATE SET
                negocio = EXCLUDED.negocio,
                pendientes = EXCLUDED.pendientes,
                fecha_limite = EXCLUDED.fecha_limite,
                adeudo = EXCLUDED.adeudo;
            """,
            (rfc, datos["negocio"], json.dumps(datos["pendientes"]), datos["fecha_limite"], datos["adeudo"]),
        )

    print(f"{len(CATALOGO_DEMO)} RFCs de ejemplo sembrados en catalogo_obligaciones.")


if __name__ == "__main__":
    main()
