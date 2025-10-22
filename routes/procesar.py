from fastapi import APIRouter, HTTPException
import pandas as pd
from ibm_client import granite_model
import re, os

router = APIRouter()

# --- Función para limpiar texto generado ---
def normalize_text(text: str) -> str:
    t = text.replace('\n', ' ').replace('\\n', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    return t

@router.post("/procesar_dataset")
async def procesar_dataset():
    """
    Lee el dataset local, usa la columna combinada ('texto_unificado') 
    para generar resúmenes y categorías con los modelos de WatsonX.
    """
    try:
        # 1️⃣ Leer dataset desde carpeta data/
        dataset_path = "data/dataset_educacion_enriquecido.csv"
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset no encontrado.")

        df = pd.read_csv(dataset_path)

        if "texto_unificado" not in df.columns:
            raise HTTPException(status_code=400, detail="No se encontró la columna 'texto_unificado'.")

        # 2️⃣ Iterar por cada texto y generar resumen y categoría
        resúmenes, categorías = [], []

        for texto in df["texto_unificado"]:
            # --- Resumen ---
            prompt_resumen = (
                f"Resume el siguiente texto en una sola frase clara y breve, "
                f"sin repetir ideas:\n\n{texto}"
            )

            params = {"decoding_method": "greedy", "max_new_tokens": 60, "temperature": 0.3}
            response_r = granite_model.generate(prompt_resumen, params=params)

            resumen = ""
            if isinstance(response_r, dict):
                resumen = response_r.get("results", [{}])[0].get("generated_text", "")
            else:
                resumen = str(response_r)
            resumen = normalize_text(resumen)
            resúmenes.append(resumen)

            # --- Clasificación ---
            prompt_categoria = (
                f"Clasifica el siguiente texto según su tipo de problema en educación: "
                f"conectividad, infraestructura o formación.\n\nTexto:\n{texto}\n\n"
                f"Responde solo con una palabra."
            )

            response_c = granite_model.generate(prompt_categoria, params=params)
            categoria = ""
            if isinstance(response_c, dict):
                categoria = response_c.get("results", [{}])[0].get("generated_text", "")
            else:
                categoria = str(response_c)
            categoria = normalize_text(categoria)
            categorías.append(categoria)

        # 3️⃣ Añadir columnas nuevas al dataset
        df["resumen_modelo"] = resúmenes
        df["categoria_modelo"] = categorías

        # 4️⃣ Guardar resultado
        salida_path = "data/dataset_procesado.csv"
        df.to_csv(salida_path, index=False)

        return {"mensaje": "Dataset procesado con éxito", "archivo_salida": salida_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando dataset: {e}")