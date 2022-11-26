from requests import JSONDecodeError
from .apps import ModelInferenceServiceConfig
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from sklearn.decomposition import PCA
from model_inference_service.constants import *
from model_inference_service.utils import transform_data, extract_data


import numpy as np

@csrf_exempt
def serve_requests(request):
    if request.method == 'GET':
        res=['Malignant', 'Benign']
        inference_data = extract_data(request)
        np.array(inference_data).reshape(-1,1)
        return HttpResponse(ModelInferenceServiceConfig.predictor(inference_data))
        # return JsonResponse(inference_data, safe=False)
    else:
        return HttpResponse('Error!')







