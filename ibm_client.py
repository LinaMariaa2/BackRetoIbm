from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import os

# Cargar variables de entorno
load_dotenv()

api_key = os.getenv("IBM_API_KEY")
project_id = os.getenv("IBM_PROJECT_ID")
url = os.getenv("IBM_URL")

# Configurar credenciales
creds = Credentials(url=url, api_key=api_key)

# Modelo base de IBM Granite
granite_model = ModelInference(
    model_id="ibm/granite-3-8b-instruct",
    credentials=creds,
    project_id=project_id
)