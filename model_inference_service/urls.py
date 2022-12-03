from django.urls import path, include
from model_inference_service.views import serve_requests


urlpatterns = [
   
    path('', serve_requests)
]
