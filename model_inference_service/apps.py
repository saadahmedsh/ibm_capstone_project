from django.apps import AppConfig
from tensorflow import keras
from model_inference_service.constants import *

class ModelInferenceServiceConfig(AppConfig):
    name = 'model_inference_service'
    predictor = None

    def ready(self):
        self.predictor = keras.models.load_model(MODEL_PATH)

        
    
