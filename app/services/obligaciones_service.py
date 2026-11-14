"""
Consulta de obligaciones pendientes de una PyME registrada en el despacho.

En el notebook, `CATALOGO_OBLIGACIONES` era un diccionario fijo en memoria
(celda 20) con 3 RFCs de ejemplo. Aqui pasa a ser una tabla real
(`catalogo_obligaciones`, ver app/database.py) para que cada despacho
pueda tener su propio catalogo real, actualizable sin tocar codigo ni
reiniciar el servidor -- ver scripts/seed_catalogo_demo.py para poblarla
con los mismos 3 registros de ejemplo del notebook, solo para pruebas.
"""

from app.database import consultar


def consultar_obligaciones_pendientes(rfc: str | None = None) -> str:
    if not rfc:
        return "Para consultar tus obligaciones pendientes necesito tu RFC, ¿me lo compartes?"

    rfc_norm = rfc.strip().upper().replace(" ", "")
    filas = consultar(
        "SELECT negocio, pendientes, fecha_limite, adeudo FROM catalogo_obligaciones WHERE rfc = %s;",
        (rfc_norm,),
    )
    if not filas:
        return f"No encontramos el RFC {rfc_norm} registrado en el despacho. Verifica que este bien escrito o contacta a tu asesor."

    negocio, pendientes, fecha_limite, adeudo = filas[0]
    if not pendientes:
        return f"{negocio}: no tienes declaraciones pendientes por el momento. Todo en orden."

    pendientes_txt = ", ".join(pendientes)
    aviso_adeudo = f" Tienes un adeudo registrado de ${float(adeudo):.2f}." if adeudo and float(adeudo) > 0 else ""
    return (
        f"{negocio}: tienes pendiente(s) {pendientes_txt}, con fecha limite el "
        f"{fecha_limite}.{aviso_adeudo}"
    )


HERRAMIENTA_OBLIGACIONES = {
    "type": "function",
    "function": {
        "name": "consultar_obligaciones_pendientes",
        "description": (
            "Consulta si una PyME registrada en el despacho tiene declaraciones "
            "fiscales pendientes, su fecha limite y si tiene algun adeudo. "
            "Requiere el RFC del negocio; si el usuario no lo ha dado, omite el parametro."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rfc": {
                    "type": "string",
                    "description": "El RFC del negocio a consultar. Omite este parametro si el usuario no dio un RFC.",
                }
            },
            "required": [],
        },
    },
}
