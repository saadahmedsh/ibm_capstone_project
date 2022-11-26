from django.apps import AppConfig
from tensorflow import keras
from model_inference_service.constants import *

class ModelInferenceServiceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'model_inference_service'
    predictor = keras.models.load_model(MODEL_PATH)
    
