# Primero las importaciones estándar de Python
import logging
import os
from typing import Dict, Any, Literal, List
from enum import Enum
import numpy as np

# Luego las importaciones de terceros
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from dotenv import load_dotenv

# Finalmente las importaciones locales
from app.utils.data_processing import ModelLoader

# Cargar variables de entorno
load_dotenv()

# Configuración de entorno
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
VERSION = os.getenv('VERSION', '1.0.0')

# Configurar logging según el ambiente
log_level = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# [Las clases Gender, EffortInput y MarathonInput se mantienen igual...]

# Inicialización de la aplicación FastAPI con más metadatos
app = FastAPI(
    title="Marathon Time Prediction API",
    description="API para predecir tiempos de maratón y calcular esfuerzo físico usando modelos de Machine Learning",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "predicción",
            "description": "Endpoints para predicción de tiempo de maratón"
        },
        {
            "name": "esfuerzo",
            "description": "Endpoints para cálculo de esfuerzo físico"
        },
        {
            "name": "sistema",
            "description": "Endpoints de monitoreo y estado del sistema"
        }
    ]
)

# Middleware CORS (solo una vez)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Especificamos solo los métodos que usamos
    allow_headers=["*"],
)

# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Inicializar el modelo con mejor manejo de errores
try:
    model_loader = ModelLoader()
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error crítico al cargar el modelo: {str(e)}")
    raise

@app.post("/predict", response_model=Dict[str, Any], tags=["predicción"])
async def predict_marathon(input_data: MarathonInput) -> Dict[str, Any]:
    """
    Predice el tiempo de maratón basado en los datos de entrada proporcionados.
    
    Args:
        input_data: Datos del corredor y condiciones de la carrera
        
    Returns:
        Dict con el tiempo predicho en horas y las unidades
        
    Raises:
        HTTPException: Si hay algún error durante la predicción
    """
    try:
        logger.debug(f"Datos recibidos: {input_data.dict()}")
        input_dict = input_data.dict()
        prediction = model_loader.predict(input_dict)
        logger.info(f"Predicción exitosa: {prediction}")
        
        return {
            "status": "success",
            "predicted_time": round(prediction, 2),
            "units": "hours",
            "timestamp": logging.Formatter().converter()
        }
    except Exception as e:
        logger.error(f"Error durante la predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/calculate-effort", response_model=Dict[str, Any], tags=["esfuerzo"])
async def calculate_effort(input_data: EffortInput) -> Dict[str, Any]:
    """
    Calcula el esfuerzo relativo y escalado basado en los datos de entrada.
    
    Args:
        input_data: Datos del ejercicio y del usuario
        
    Returns:
        Dict con el esfuerzo relativo y escalado
    """
    try:
        logger.debug(f"Datos recibidos para cálculo de esfuerzo: {input_data.dict()}")
        
        # [El resto del código se mantiene igual...]
        
        return {
            "status": "success",
            "esfuerzo_relativo": round(float(esfuerzo_relativo), 2),
            "esfuerzo_escalado": round(esfuerzo_escalado, 2),
            "timestamp": logging.Formatter().converter(),
            "detalles": {
                "factor_sexo": factor_sexo,
                "factor_edad": round(factor_edad, 2),
                "esfuerzo_zonas": round(float(esfuerzo_zonas), 2),
                "proporcion_fc": round(float(proporcion_fc), 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error durante el cálculo del esfuerzo: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en el cálculo: {str(e)}"
        )

@app.get("/health", tags=["sistema"])
async def health_check():
    """
    Endpoint para verificar el estado de la API.
    
    Returns:
        Dict con el estado del servicio
    """
    try:
        return {
            "status": "healthy",
            "environment": ENVIRONMENT,
            "version": VERSION,
            "timestamp": logging.Formatter().converter(),
            "debug_mode": DEBUG,
            "model_info": {
                "type": str(type(model_loader.model)),
                "scaler": str(type(model_loader.scaler)),
                "loaded": True
            },
            "system_info": {
                "python_version": os.sys.version,
                "api_host": API_HOST,
                "api_port": API_PORT
            }
        }
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en health check: {str(e)}"
        )

# Manejador de errores personalizado mejorado
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": str(exc),
            "path": str(request.url),
            "timestamp": logging.Formatter().converter(),
            "environment": ENVIRONMENT
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info"
    )