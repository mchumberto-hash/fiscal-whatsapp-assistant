"""
Herramienta de desarrollo: recalibra UMBRAL_BASE_FISCAL con el set de
evaluacion de recall@k / tasa de falso positivo. Equivale a las celdas 17,
18c y 18d del notebook original.

NO es parte de la app en produccion -- usa numpy en memoria (rapido,
no toca Supabase) a proposito, igual que en el notebook: es una
herramienta de desarrollo para re-calibrar el umbral cada vez que se edite
`DOCUMENTOS` en scripts/migrar_base_fiscal.py, no el camino que usa el bot
real (ese es app/services/rag_service.py, contra pgvector).

Correr con: python -m scripts.evaluar_umbral
Despues de correrlo, actualizar UMBRAL_BASE_FISCAL en la configuracion
(variable de entorno, o el default en app/config.py) con el valor reportado.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.migrar_base_fiscal import DOCUMENTOS

# (pregunta, indice_del_documento_esperado_en DOCUMENTOS, o None si es fuera de alcance)
PREGUNTAS_EVALUACION = [
    ("¿Cuáles son las tasas de ISR en RESICO?", 0),
    ("¿Hasta cuánto puedo ganar al año para estar en RESICO?", 0),
    ("Si cobro 60 mil pesos en un mes en RESICO, ¿qué tasa de ISR me aplica?", 0),

    ("¿Cuándo se presenta la declaración anual de personas físicas?", 1),
    ("¿Hasta qué día tengo para presentar mi declaración anual?", 1),
    ("¿Dónde presento mi declaración anual, en qué portal?", 2),

    ("¿Cuándo tengo que presentar mi pago provisional mensual?", 3),
    ("¿Hasta qué día del mes puedo declarar mis pagos provisionales?", 3),

    ("¿Cuánto es la tasa de IVA?", 4),
    ("¿Cuál es la tasa de IVA en la frontera?", 4),
    ("¿Qué productos tienen tasa 0% de IVA?", 4),

    ("¿Cuál es el límite de las deducciones personales?", 5),
    ("¿Hasta cuánto puedo deducir en total entre todos mis gastos personales?", 5),

    ("¿Hasta cuándo puedo deducir mis gastos médicos?", 6),
    ("¿Puedo deducir una consulta de psicología?", 6),

    ("¿Si me hospitalizan, ese gasto es deducible?", 7),

    ("¿Puedo deducir mis lentes?", 8),

    ("¿Puedo deducir la colegiatura de mis hijos?", 9),
    ("¿Hay tope para deducir la escuela de mi hijo en primaria?", 9),

    ("¿Los donativos que hago se pueden deducir?", 10),
    ("¿Cuánto puedo deducir de donativos a una asociación civil?", 10),

    ("¿Puedo deducir los intereses de mi crédito hipotecario?", 11),

    ("¿Los gastos funerarios de mi papá son deducibles?", 12),

    ("¿Las aportaciones que hago para mi retiro son deducibles?", 13),

    ("¿Qué necesito para hacer una factura (CFDI)?", 14),
    ("¿Qué datos del cliente necesito para facturar correctamente?", 14),

    ("Si pago tarde una declaración, ¿cuánto me cobran de recargos?", 15),
    ("¿Cuánto aumenta mi deuda si pido pagar en parcialidades al SAT?", 15),

    ("¿Qué necesito para sacar mi RFC?", 16),
    ("¿Cómo me inscribo al RFC como persona física con actividad empresarial?", 16),

    # Fuera de alcance: no hay documento correcto, debe regresar el aviso de "no sé"
    ("¿Cómo tributan los ingresos por streaming en plataformas extranjeras?", None),
    ("¿Cómo declaro ganancias de criptomonedas?", None),
    ("¿Cuánto tengo que pagar de impuesto sobre una herencia?", None),
    ("¿Cómo le hago para dar de baja a un trabajador del IMSS?", None),
    ("¿Qué pasa si vendo mi casa, pago impuestos?", None),
    ("¿Cuánto tiempo tengo para renovar mi credencial del INE?", None),
]


def main() -> None:
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings_documentos = modelo.encode(DOCUMENTOS, normalize_embeddings=True)

    def similitudes(pregunta):
        embedding_pregunta = modelo.encode([pregunta], normalize_embeddings=True)
        return np.dot(embeddings_documentos, embedding_pregunta.T).flatten()

    def evaluar_umbral(umbral, k=2):
        aciertos_alcance, total_alcance = 0, 0
        falsos_positivos, total_fuera = 0, 0
        fallos_detalle = []

        for pregunta, doc_esperado in PREGUNTAS_EVALUACION:
            sims = similitudes(pregunta)
            indices_topk = np.argsort(sims)[::-1][:k]
            relevantes = [i for i in indices_topk if sims[i] >= umbral]

            if doc_esperado is not None:
                total_alcance += 1
                if doc_esperado in relevantes:
                    aciertos_alcance += 1
                else:
                    fallos_detalle.append((pregunta, "deberia_recuperar", doc_esperado, sims[doc_esperado]))
            else:
                total_fuera += 1
                if relevantes:
                    falsos_positivos += 1
                    fallos_detalle.append((pregunta, "falso_positivo", relevantes[0], sims[relevantes[0]]))

        recall = aciertos_alcance / total_alcance if total_alcance else float("nan")
        tasa_fp = falsos_positivos / total_fuera if total_fuera else float("nan")
        return recall, tasa_fp, fallos_detalle

    mejor_umbral, mejor_score, mejor_fallos = None, -1, None
    print(f"{'umbral':>7} | {'recall_en_alcance':>17} | {'tasa_falso_positivo':>19}")
    for umbral in np.arange(0.20, 0.81, 0.02):
        recall, fp, fallos = evaluar_umbral(round(float(umbral), 2))
        print(f"{umbral:7.2f} | {recall:17.0%} | {fp:19.0%}")
        score = recall - fp
        if score > mejor_score:
            mejor_umbral, mejor_score, mejor_fallos = round(float(umbral), 2), score, fallos

    print(f"\nMejor umbral encontrado: {mejor_umbral}")
    print("Casos que ese umbral sigue fallando:")
    for fallo in mejor_fallos:
        print(" ", fallo)
    print(f"\nActualiza UMBRAL_BASE_FISCAL (app/config.py o la variable de entorno) a {mejor_umbral}.")


if __name__ == "__main__":
    main()
