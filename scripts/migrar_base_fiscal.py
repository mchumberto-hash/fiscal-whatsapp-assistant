"""
Script de un solo uso: genera los embeddings de la base de conocimiento
fiscal y los migra a la tabla `documentos_fiscales` en Supabase/pgvector.

Equivale a las celdas 16 + 18f del notebook original. Se corre a mano
(`python -m scripts.migrar_base_fiscal`) cada vez que el contenido de
`DOCUMENTOS` cambie -- por ejemplo, al curar contenido fiscal real para un
despacho especifico en vez de los documentos de ejemplo de aqui.

Usa TRUNCATE antes de insertar (igual que el notebook), para que volver a
correrlo despues de editar `DOCUMENTOS` deje la tabla sincronizada sin
arrastrar filas viejas de documentos que ya no existen o cambiaron de indice.

IMPORTANTE: si se edita `DOCUMENTOS`, hay que volver a correr
`scripts/evaluar_umbral.py` y actualizar UMBRAL_BASE_FISCAL en la
configuracion (ver app/config.py) con el resultado -- el umbral actual
(0.56) esta calibrado especificamente para este contenido.
"""

import psycopg2
from sentence_transformers import SentenceTransformer

from app.config import SUPABASE_DB_URL

# Mismo contenido y mismas fuentes citadas que el notebook (celda 16).
# ⚠️ Datos fiscales ilustrativos para el ejercicio 2026 -- un contador
# certificado debe validar cada cifra contra el texto oficial (Ley de
# Ingresos de la Federacion 2026, RMF 2026, sat.gob.mx) antes de usarlos
# con un cliente real de un despacho.
DOCUMENTOS = [
    # 0: RESICO
    "El Régimen Simplificado de Confianza (RESICO) para personas físicas aplica a quienes "
    "tienen ingresos anuales de hasta 3.5 millones de pesos por actividad empresarial, "
    "profesional o arrendamiento (este límite es de referencia: si se excede, se debe salir "
    "del régimen). El ISR se paga de forma simplificada, sin deducir gastos, con tasas "
    "progresivas sobre los ingresos efectivamente cobrados: 1.00% hasta $25,000 mensuales, "
    "1.10% hasta $50,000, 1.50% hasta $83,333.33, 2.00% hasta $208,333.33, y 2.50% para el "
    "resto hasta el límite anual de $3.5 millones.",

    # 1: Declaración anual — fecha límite
    "La declaración anual de personas físicas se presenta a más tardar el 30 de abril. "
    "Presentarla después del 30 de abril puede generar una multa.",

    # 2: Declaración anual — dónde se presenta
    "La declaración anual de personas físicas se presenta en el Portal del SAT (sat.gob.mx) "
    "o en la app Declaración Anual Personas Físicas.",

    # 3: Pago provisional
    "Las personas físicas con actividad empresarial o profesional deben presentar su pago "
    "provisional (declaración mensual) a más tardar el día 17 del mes siguiente al que se "
    "declara, a través del Portal del SAT.",

    # 4: IVA
    "La tasa general de IVA en México es del 16% sobre la mayoría de bienes y servicios. "
    "En la región fronteriza norte y sur existe un estímulo que reduce la tasa al 8% para "
    "contribuyentes inscritos en el padrón de beneficiarios (vigente hasta el 31 de "
    "diciembre de 2026). Existe una tasa del 0% para ciertos productos como alimentos "
    "básicos y medicinas, y algunos servicios están exentos de IVA.",

    # 5: Deducciones personales — límite general
    "Las deducciones personales para personas físicas (como gastos médicos, colegiaturas o "
    "donativos) están limitadas al monto que resulte menor entre 5 UMAs anuales "
    "(aproximadamente $206,418.60 para 2026) o el 15% del total de los ingresos del "
    "contribuyente. Este es el tope general que aplica a la suma de todas las deducciones "
    "personales juntas.",

    # 6: Deducciones — consultas médicas, dentales y psicológicas
    "Las consultas médicas, dentales, de psicología y de nutrición son gastos deducibles, "
    "igual que los análisis clínicos.",

    # 7: Deducciones — hospitalización y seguros
    "Los gastos de hospitalización y las primas de seguros de gastos médicos mayores son "
    "gastos deducibles.",

    # 8: Deducciones — lentes
    "Los lentes ópticos son deducibles, con un tope de $2,500 anuales.",

    # 9: Deducciones — colegiaturas y transporte escolar
    "Las colegiaturas son deducibles con un tope anual por nivel educativo: preescolar "
    "$14,200, primaria $12,900, secundaria $19,900, profesional técnico $17,100 y "
    "bachillerato $24,500 (la escuela debe tener RVOE). El transporte escolar solo es "
    "deducible si la escuela lo exige como obligatorio.",

    # 10: Deducciones — donativos
    "Los donativos que se hacen a donatarias autorizadas son deducibles hasta el 7% de los "
    "ingresos acumulables del ejercicio anterior, con un 4% adicional disponible si el "
    "donativo es para gobierno federal, estatal o municipal.",

    # 11: Deducciones — intereses hipotecarios
    "Los intereses reales de un crédito hipotecario para casa habitación son deducibles, "
    "si el crédito no es mayor a 750,000 UDIs.",

    # 12: Deducciones — gastos funerarios
    "Los gastos funerarios de familiares directos son deducibles, con un tope de 1 UMA "
    "(aproximadamente $41,283.72).",

    # 13: Deducciones — aportaciones voluntarias al retiro
    "Las aportaciones voluntarias para el retiro son deducibles, con un tope del 10% de "
    "los ingresos acumulables o 5 UMAs anuales, lo que resulte menor.",

    # 14: CFDI 4.0
    "Para facturar (emitir un CFDI 4.0) necesitas los datos del cliente o receptor: su RFC "
    "y su nombre completo, tal como aparecen en su constancia de situación fiscal. También "
    "necesitas el uso de CFDI y la forma de pago.",

    # 15: Recargos
    "Si una declaración o pago se presenta fuera del plazo establecido, el SAT cobra "
    "recargos mensuales sobre la contribución omitida: una tasa de mora del 0.98% mensual "
    "sobre saldos insolutos según la Ley de Ingresos de la Federación 2026, más la "
    "actualización correspondiente. Si el contribuyente solicita pago a plazos (parcialidades "
    "o diferido), la tasa mensual es mayor: 1.26% hasta 12 meses, 1.53% de 13 a 24 meses, y "
    "1.82% a más de 24 meses o pago diferido. Además puede aplicar multas adicionales.",

    # 16: Inscripción al RFC
    "Para sacar tu RFC o inscribirte como persona física con actividad empresarial, agenda "
    "una cita en el Portal del SAT y presenta tu identificación oficial vigente, tu CURP y "
    "un comprobante de domicilio no mayor a 3 meses de antigüedad.",
]

# Fuentes consultadas (nov-2026) -- verificar contra texto oficial antes de produccion:
# - RESICO 2026: https://www.satfacil.com.mx/blog/regimen-simplificado-confianza-resico-guia-2026
# - Declaración anual, fecha límite: https://sdv.com.mx/recursos/declaracion-anual-personas-fisicas-2026/
# - IVA general/fronterizo/0%: https://commoner-law.com/mexico/derechos-fiscales/iva
# - Deducciones personales 2026: https://www.elcontribuyente.mx/deducciones-personales/
# - Recargos 2026, LIF art. 5: https://sdv.com.mx/compendio/ley-ingresos-2026/articulo-5/


def main() -> None:
    print("Cargando modelo de embeddings...")
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = modelo.encode(DOCUMENTOS, normalize_embeddings=True)
    print("Embeddings generados:", embeddings.shape)

    conexion = psycopg2.connect(SUPABASE_DB_URL)
    conexion.autocommit = True
    cursor = conexion.cursor()

    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    dim = embeddings.shape[1]
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS documentos_fiscales (
            id INTEGER PRIMARY KEY,
            contenido TEXT NOT NULL,
            embedding vector({dim})
        );
    """)

    cursor.execute("TRUNCATE TABLE documentos_fiscales;")
    for i, (contenido, vector) in enumerate(zip(DOCUMENTOS, embeddings)):
        vector_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
        cursor.execute(
            "INSERT INTO documentos_fiscales (id, contenido, embedding) VALUES (%s, %s, %s::vector)",
            (i, contenido, vector_str),
        )

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documentos_fiscales_embedding
        ON documentos_fiscales USING hnsw (embedding vector_cosine_ops);
    """)

    print(f"{len(DOCUMENTOS)} documentos migrados a Supabase/pgvector (dimension {dim}).")


if __name__ == "__main__":
    main()
