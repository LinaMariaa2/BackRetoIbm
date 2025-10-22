from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ibm_client import granite_model
import re

# Inicializar router
router = APIRouter()

# Modelo de entrada
class TextoEntrada(BaseModel):
    texto: str

# --- Utilidad: limpiar texto generado ---
def normalize_text(text: str) -> str:
    """Limpia texto de saltos de línea y espacios extra."""
    t = text.replace('\n', ' ').replace('\\n', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    return t


# --- Endpoint principal ---
@router.post("/resumir")
async def resumir_texto(data: TextoEntrada):
    """
    Genera un resumen corto, claro y limpio de un texto usando IBM Watsonx.ai.
    """
    try:
        # 1️⃣ Crear prompt con instrucciones claras
        prompt = (
            f"Resume el siguiente texto en una sola frase clara y breve, "
            f"sin repetir ideas ni agregar información adicional:\n\n{data.texto}"
        )

        # 2️⃣ Parámetros del modelo (control de creatividad y longitud)
        params = {
            "decoding_method": "greedy",
            "max_new_tokens": 60,   # aumenta un poco si los resúmenes salen cortos
            "min_new_tokens": 10,
            "temperature": 0.3
        }

        # 3️⃣ Llamada al modelo Watsonx
        response = granite_model.generate(prompt, params=params)

        # 4️⃣ Normalización del formato de salida
        if isinstance(response, dict):
            raw_text = response.get("results", [{}])[0].get("generated_text", "")
        else:
            raw_text = str(response)

        # 5️⃣ Limpieza del texto
        resumen = normalize_text(raw_text)

        # 6️⃣ Control de longitud (máx. 30 palabras)
        palabras = resumen.split()
        if len(palabras) > 30:
            resumen = " ".join(palabras[:30]) + "..."

        # 7️⃣ Retornar resultado
        return {
            "input_text": data.texto,
            "resumen": resumen
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al generar el resumen: {str(e)}"
        )