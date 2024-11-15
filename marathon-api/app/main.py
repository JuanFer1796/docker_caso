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

class Gender(str, Enum):
    hombre = "hombre"
    mujer = "mujer"

class EffortInput(BaseModel):
    edad: int
    sexo: Gender
    peso: float
    frecuencia_cardiaca_reposo: int
    frecuencia_cardiaca_maxima: int
    frecuencia_cardiaca_promedio: int
    tiempo_zona_1: float
    tiempo_zona_2: float
    tiempo_zona_3: float
    tiempo_zona_4: float
    tiempo_zona_5: float

    @validator('edad')
    def validate_edad(cls, v):
        if v < 18 or v > 100:
            raise ValueError('La edad debe estar entre 18 y 100 años')
        return v

    @validator('peso')
    def validate_peso(cls, v):
        if v <= 0 or v > 200:
            raise ValueError('El peso debe estar entre 0 y 200 kg')
        return v

    @validator('frecuencia_cardiaca_reposo')
    def validate_fcr(cls, v):
        if v < 40 or v > 120:
            raise ValueError('La frecuencia cardíaca en reposo debe estar entre 40 y 120 bpm')
        return v

    @validator('frecuencia_cardiaca_maxima')
    def validate_fcm(cls, v, values):
        if 'frecuencia_cardiaca_reposo' in values:
            if v <= values['frecuencia_cardiaca_reposo']:
                raise ValueError('La FCM debe ser mayor que la FC en reposo')
        if v < 100 or v > 220:
            raise ValueError('La frecuencia cardíaca máxima debe estar entre 100 y 220 bpm')
        return v

    @validator('frecuencia_cardiaca_promedio')
    def validate_fcp(cls, v, values):
        if 'frecuencia_cardiaca_reposo' in values and 'frecuencia_cardiaca_maxima' in values:
            if v < values['frecuencia_cardiaca_reposo'] or v > values['frecuencia_cardiaca_maxima']:
                raise ValueError('La FC promedio debe estar entre la FC reposo y la FC máxima')
        return v

    @validator('tiempo_zona_1', 'tiempo_zona_2', 'tiempo_zona_3', 'tiempo_zona_4', 'tiempo_zona_5')
    def validate_tiempos(cls, v):
        if v < 0:
            raise ValueError('Los tiempos en zona deben ser positivos')
        return v

class MarathonInput(BaseModel):
    AGE: int
    RunType: Literal['Outdoor', 'Indoor']
    SubTime: float
    SubDistance: float
    Wall21: float
    km4week: float
    sp4week: float
    CrossTraining: str
    Wall21_Marathon: float
    PRECIP_mm: float
    SUNSHINE_hrs: float
    CLOUD_hrs: float
    ATMOS_PRESS_mbar: float
    AVG_TEMP_C: float
    MAX_TEMP_C: float
    MIN_TEMP_C: float
    GENDER: int  # 0 o 1

    @validator('AGE')
    def validate_age(cls, v):
        if v < 18 or v > 100:
            raise ValueError('La edad debe estar entre 18 y 100 años')
        return v

    @validator('CrossTraining')
    def validate_cross_training(cls, v):
        if not v.endswith('h'):
            raise ValueError('CrossTraining debe terminar con "h" (ejemplo: "5h")')
        try:
            hours = float(v[:-1])
            if hours < 0 or hours > 100:
                raise ValueError('Las horas de CrossTraining deben estar entre 0 y 100')
        except ValueError:
            raise ValueError('Formato inválido para CrossTraining')
        return v

    @validator('GENDER')
    def validate_gender(cls, v):
        if v not in [0, 1]:
            raise ValueError('GENDER debe ser 0 o 1')
        return v

    @validator('SubTime')
    def validate_subtime(cls, v):
        if v <= 0:
            raise ValueError('SubTime debe ser mayor que 0')
        return v

    @validator('SubDistance')
    def validate_subdistance(cls, v):
        if v <= 0:
            raise ValueError('SubDistance debe ser mayor que 0')
        return v

    @validator('Wall21')
    def validate_wall21(cls, v):
        if v <= 0:
            raise ValueError('Wall21 debe ser mayor que 0')
        return v

    @validator('km4week')
    def validate_km4week(cls, v):
        if v < 0:
            raise ValueError('km4week no puede ser negativo')
        return v

    @validator('sp4week')
    def validate_sp4week(cls, v):
        if v < 0:
            raise ValueError('sp4week no puede ser negativo')
        return v

    @validator('ATMOS_PRESS_mbar')
    def validate_pressure(cls, v):
        if v < 800 or v > 1200:
            raise ValueError('La presión atmosférica debe estar entre 800 y 1200 mbar')
        return v

    @validator('AVG_TEMP_C')
    def validate_avg_temp(cls, v, values):
        if 'MIN_TEMP_C' in values and 'MAX_TEMP_C' in values:
            if not (values['MIN_TEMP_C'] <= v <= values['MAX_TEMP_C']):
                raise ValueError('La temperatura promedio debe estar entre la mínima y la máxima')
        return v

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

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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
    try:
        logger.debug(f"Datos recibidos para cálculo de esfuerzo: {input_data.dict()}")
        
        # Pesos para cada zona
        pesos_zonas = [1, 2, 4, 8, 16]
        
        # Definir el factor de sexo
        factor_sexo = 1 if input_data.sexo == Gender.hombre else 0.9
        
        # Calcular el factor de edad
        factor_edad = 1 - 0.02 * ((input_data.edad - 30) / 10)
        
        # Calcular el esfuerzo relativo en base a tiempo en zona y pesos
        esfuerzo_zonas = (
            input_data.tiempo_zona_1 * pesos_zonas[0] +
            input_data.tiempo_zona_2 * pesos_zonas[1] +
            input_data.tiempo_zona_3 * pesos_zonas[2] +
            input_data.tiempo_zona_4 * pesos_zonas[3] +
            input_data.tiempo_zona_5 * pesos_zonas[4]
        )
        
        # Calcular la proporción de la frecuencia cardíaca
        rango_fc = input_data.frecuencia_cardiaca_maxima - input_data.frecuencia_cardiaca_reposo
        proporcion_fc = (input_data.frecuencia_cardiaca_promedio - input_data.frecuencia_cardiaca_reposo) / rango_fc
        
        # Calcular el esfuerzo relativo total
        esfuerzo_relativo = esfuerzo_zonas * factor_sexo * factor_edad * proporcion_fc
        
        # Escalar con una sigmoide para que esté entre 0 y 100
        esfuerzo_escalado = float(1 / (1 + np.exp(-esfuerzo_relativo / 100)) * 100)
        
        logger.debug(f"Esfuerzo calculado - Relativo: {esfuerzo_relativo}, Escalado: {esfuerzo_escalado}")
        
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