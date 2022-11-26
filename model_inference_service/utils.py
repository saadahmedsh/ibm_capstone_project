from sklearn.preprocessing import MinMaxScaler
from model_inference_service.constants import *
from sklearn.decomposition import PCA


def extract_data(req):
    features = req.GET.keys()
    vals =  []
    for key in features:
        vals.append(float(req.GET[key]))
    return vals

def transform_data(req):
    scaler = MinMaxScaler()
    data=extract_data(req)
    data_rescaled= scaler.fit_transform(data)
    print('Type=============', type(data_rescaled))
    components = PCA(n_components=NO_OF_COMPONENTS).fit_transform(data_rescaled.reshape(-1,1))
    return components