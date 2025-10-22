from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importar los routers
from routes import resumen, clasificar, procesar

app = FastAPI(
    title="Reto IBM Senasoft 2025 🚀",
    description="API que utiliza modelos de IBM Watsonx.ai para resumen, clasificación y procesamiento de datos educativos.",
    version="1.0.0"
)

# Configuración de CORS
origins = [
    "http://localhost:3000",   # frontend local
    "https://tu-front.app"     # URL de despliegue (cuando tengas hosting)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar los routers
app.include_router(resumen.router, prefix="/api", tags=["🧠 Resumen"])
app.include_router(clasificar.router, prefix="/api", tags=["🏷️ Clasificación"])
app.include_router(procesar.router, prefix="/api", tags=["📊 Procesamiento Dataset"])

# Endpoint raíz
@app.get("/")
def root():
    return {
        "mensaje": "✅ API del Reto IBM Senasoft funcionando correctamente.",
        "rutas_disponibles": [
            "/api/resumir",
            "/api/clasificar",
            "/api/procesar_dataset"
        ]
    }
