# from .apps import ModelInferenceServiceConfig
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from sklearn.decomposition import PCA
from model_inference_service.constants import *
from model_inference_service.utils import *

import tensorflow as tf
import numpy as np


# One time initilization
model = tf.keras.models.load_model(MODEL_PATH)

@csrf_exempt
def serve_requests(request):
    if request.method == GET:
        inference_data = transform_data(request)
        pred = model.predict(inference_data)
        if pred >= 0.5:
            return HttpResponse(LABELS[0])
        else:
            return HttpResponse(LABELS[1])
    else:
        return HttpResponse('Error!')


