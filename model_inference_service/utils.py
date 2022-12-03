from sklearn.preprocessing import MinMaxScaler
from model_inference_service.constants import *
from sklearn.decomposition import PCA
from .constants import *
import numpy as np
import keras
import pickle as pk


pca = pk.load(open(PCA_PATH,'rb'))

def load_model():
    return keras.models.load_model(MODEL_PATH)


def extract_data(req):
    features = req.GET.keys()
    vals =  [float(req.GET[key]) for key in features]
    return np.array(vals)

def transform_data(req):
    scaler = MinMaxScaler()
    data= np.array(extract_data(req))
    data_rescaled= scaler.fit_transform(data.reshape(-1,1))
    components = pca.transform(np.expand_dims(data_rescaled.reshape(1,-1)[0], axis=0))
    return components