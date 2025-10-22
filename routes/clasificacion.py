from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ibm_client import granite_model

router = APIRouter()

class TextoEntrada(BaseModel):
    texto: str

@router.post("/clasificar")
async def clasificar_texto(data: TextoEntrada):
    try:
        prompt = f"""
Analiza el siguiente texto y clasifícalo en una de estas categorías:
- Educación
- Seguridad
- Salud
- Medio Ambiente

Texto: "{data.texto}"

Responde solo con una palabra: la categoría.
"""

        params = {
            "decoding_method": "greedy",
            "max_new_tokens": 20,
            "temperature": 0.0
        }

        response = granite_model.generate_text(prompt=prompt, params=params)

        if isinstance(response, dict):
            categoria = response.get("results", [{}])[0].get("generated_text", "").strip()
        else:
            categoria = str(response).strip()

        # Limpiar salida
        categoria = categoria.replace("Respuesta:", "").replace("Categoría:", "").strip()

        posibles = ["Educación", "Seguridad", "Salud", "Medio Ambiente"]
        for p in posibles:
            if p.lower() in categoria.lower():
                categoria = p
                break

        return {"categoria": categoria}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")