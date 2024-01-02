import numpy as np
import joblib
from importlib_resources import files as resource_files
from molSimplify.python_nn.tf_ANN import (tf_ANN_excitation_prepare,
                                          load_normalization_data,
                                          data_normalize,
                                          data_rescale, get_key)


def load_sklearn_model(predictor):
    key = get_key(predictor, suffix="model")
    modelfile = resource_files("molSimplify").joinpath(f"sklearn_models/{key}.h5")
    loaded_model = joblib.load(modelfile)
    return loaded_model


def sklearn_supervisor(predictor, descriptors, descriptor_names, debug=False):
    print(('scikitlearn models activated for ' + str(predictor)))
    excitation = tf_ANN_excitation_prepare(predictor, descriptors, descriptor_names)
    if debug:
        print(('excitation is ' + str(excitation.shape)))
        print('fetching non-dimensionalization data... ')
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)
    if debug:
        print('rescaling input excitation...')
    excitation = data_normalize(excitation, train_mean_x, train_var_x)
    loaded_model = load_sklearn_model(predictor)
    if "clf" not in predictor:
        result = data_rescale(loaded_model.predict(excitation), train_mean_y, train_var_y)
    else:
        result = loaded_model.predict_proba(excitation)
        result = np.array([[1 - x[0]] if x[0] >= x[1] else [x[1]] for x in result])
    model_uncertainty = [-1]
    return result, model_uncertainty
