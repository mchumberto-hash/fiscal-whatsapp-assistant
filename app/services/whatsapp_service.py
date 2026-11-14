"""
Envio de mensajes y normalizacion de numero, identico al notebook
(celdas 12 y 31) -- esta parte no tenia nada que cambiar al salir de Colab.
"""

import requests

from app.config import PHONE_ID, WHATSAPP_TOKEN
from app.services.metrics_service import registrar_evento


def normalizar_numero_mx(numero: str) -> str:
    if numero.startswith("521") and len(numero) == 13:
        return "52" + numero[3:]
    return numero


def enviar_respuesta(numero: str, texto: str) -> None:
    url = f"https://graph.facebook.com/v25.0/{PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": numero,
        "type": "text",
        "text": {"body": texto},
    }

    r = requests.post(url, headers=headers, json=payload)
    registrar_evento("respuesta_enviada", numero, {"status": r.status_code, "texto": r.text})
    r.raise_for_status()
