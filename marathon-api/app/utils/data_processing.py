import pandas as pd
import joblib
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PATH = Path(os.getenv('MODEL_PATH', 'models/'))

class ModelLoader:
    def __init__(self):
        try:
            self.model = joblib.load(MODEL_PATH / 'ridge_model.joblib')
            self.scaler = joblib.load(MODEL_PATH / 'scaler.joblib')
            self.runtype_mapping = {
                'Outdoor': 1,
                'Indoor': 0
            }
            logger.info("ModelLoader inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar ModelLoader: {e}")
            raise

    def preprocess_data(self, input_data: dict) -> pd.DataFrame:
        try:
            logger.debug(f"Iniciando preprocesamiento de datos: {input_data}")
            
            # Crear DataFrame inicial
            input_df = pd.DataFrame([input_data])
            
            # Mapear RunType
            input_df['RunType'] = input_df['RunType'].map(self.runtype_mapping)
            
            # Procesar CrossTraining
            input_df['CrossTraining'] = input_df['CrossTraining'].str.extract(r'(\d+)').astype(float).fillna(0)
            
            # GENDER_male es directamente el valor de GENDER
            input_df['GENDER_male'] = input_df['GENDER']
            
            # Ordenar columnas
            numeric_columns = [
                'AGE', 'RunType', 'SubTime', 'SubDistance', 'Wall21', 
                'km4week', 'sp4week', 'CrossTraining', 'Wall21_Marathon',
                'PRECIP_mm', 'SUNSHINE_hrs', 'CLOUD_hrs', 'ATMOS_PRESS_mbar',
                'AVG_TEMP_C', 'MAX_TEMP_C', 'MIN_TEMP_C', 'GENDER_male'
            ]
            
            # Seleccionar y ordenar columnas
            final_df = input_df[numeric_columns]
            
            logger.debug(f"Columnas finales: {final_df.columns.tolist()}")
            logger.debug(f"Datos preprocesados:\n{final_df.head()}")
            
            return final_df
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}", exc_info=True)
            raise Exception(f"Error en preprocesamiento: {str(e)}")

    def predict(self, input_data: dict) -> float:
        try:
            processed_data = self.preprocess_data(input_data)
            logger.debug(f"Datos procesados shape: {processed_data.shape}")
            logger.debug(f"Columnas procesadas: {processed_data.columns.tolist()}")
            
            scaled_data = self.scaler.transform(processed_data)
            logger.debug(f"Datos escalados shape: {scaled_data.shape}")
            
            prediction = float(self.model.predict(scaled_data)[0])
            logger.debug(f"Predicción: {prediction}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}", exc_info=True)
            raise Exception(f"Error en predicción: {str(e)}")