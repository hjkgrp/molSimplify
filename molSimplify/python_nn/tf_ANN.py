# Written by JP Janet for HJK Group
# Dpt of Chemical Engineering, MIT

##########################################################
######## This script contains a neural network  ##########
#####  trained on octahedral metal-ligand          #######
########   bond distances and spin propensity  ###########
##########################################################


import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import scipy
from typing import List, Tuple, Union, Optional
from tensorflow.keras import backend as K
from importlib_resources import files as resource_files
from packaging import version
import tensorflow as tf
import sys, types

from molSimplify.python_nn.clf_analysis_tool import array_stack, get_layer_outputs, dist_neighbor, get_entropy

import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Keras (for non-clf models and legacy fallbacks)
try:
    from keras.models import load_model, model_from_json
    from keras.optimizers import Adam
except Exception:
    from tensorflow.keras.models import load_model, model_from_json

# Optional: package data helper
try:
    from importlib.resources import files as resource_files
except Exception:
    resource_files = None



def get_layer_outputs2(model, layer_index, input_array, training_flag=False):
    """
    Return the output of model.layers[layer_index] for a single example.
    Ensures input has correct dtype/shape and matches the model's expected structure.
    """
    # 1) Build a partial model from the original graph
    try:
        # Functional/Sequential both support these
        partial_model = tf.keras.Model(inputs=model.inputs,
                                       outputs=model.layers[layer_index].output)
    except Exception:
        # Fallback for very old Sequential APIs
        partial_model = tf.keras.Model(inputs=model.input,
                                       outputs=model.layers[layer_index].output)

    # 2) Prepare inputs (ndarray float32, correct rank)
    x = _prepare_inputs_for_model(model, input_array)

    # 3) Call with the correct structure (no extra list wrappers, no strings)
    out = partial_model(x, training=training_flag)
    # Ensure eager value
    return out.numpy() if hasattr(out, "numpy") else np.array(out)


def _prepare_inputs_for_model(model, X):
    """Ensure numeric ndarray and correct rank for model.input(s)."""
    X = np.asarray(X, dtype=np.float32)
    X = _fix_shape_for_model(model, X)  # your helper: adds (time) axis if needed
    return X


def _install_sklearn_compat_shims():
    """
    Create backwards-compat import paths so old pickles load under modern sklearn.
    Covers:
      - sklearn.svm.classes -> sklearn.svm._classes
      - sklearn.externals.joblib -> joblib
      - sklearn.grid_search -> sklearn.model_selection
      - sklearn.cross_validation -> sklearn.model_selection
      - sklearn.utils.fixes (fallback to `types.SimpleNamespace`) if referenced
    """
    # 1) svm.classes -> svm._classes
    try:
        import sklearn.svm._classes as _svm_new
        mod_name = "sklearn.svm.classes"
        if mod_name not in sys.modules:
            shim = types.ModuleType(mod_name)
            for name in ("SVC","NuSVC","SVR","NuSVR","OneClassSVM","LinearSVC","LinearSVR"):
                if hasattr(_svm_new, name):
                    setattr(shim, name, getattr(_svm_new, name))
            sys.modules[mod_name] = shim
    except Exception:
        pass

    # 2) externals.joblib -> joblib
    try:
        import joblib as _jl
        mod_name = "sklearn.externals.joblib"
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _jl
    except Exception:
        pass

    # 3) grid_search -> model_selection
    try:
        import sklearn.model_selection as _ms
        mod_name = "sklearn.grid_search"
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _ms
    except Exception:
        pass

    # 4) cross_validation -> model_selection
    try:
        import sklearn.model_selection as _ms2
        mod_name = "sklearn.cross_validation"
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _ms2
    except Exception:
        pass

    # 5) utils.fixes sometimes referenced; provide a harmless placeholder
    mod_name = "sklearn.utils.fixes"
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)


def perform_ANN_prediction(RAC_dataframe: pd.DataFrame, predictor_name: str,
                           RAC_column: str = 'RACs') -> pd.DataFrame:
    # Performs a correctly normalized/rescaled prediction for a property specified by predictor_name.
    # Also calculates latent vector and smallest latent distance from training data.
    # RAC_dataframe can contain anything (e.g. a database pull) as long as it also contains the required RAC features.
    # Predictor_name can be a name like ls_ii, hs_iii, homo, oxo, hat, etc.
    # Input dataframe must have all RAC features in individual columns, or as dictionaries in a single column specified by `RAC_column`.
    # Will not execute if RAC features are missing.

    # Returns: RAC_dataframe with new columns added:
    # - predictor_name_latent_vector
    # - predictor_name_min_latent_distance,
    # - predictor_name_prediction

    assert type(RAC_dataframe) is pd.DataFrame
    train_vars = load_ANN_variables(predictor_name)
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor_name)
    my_ANN = load_keras_ann(predictor_name)

    # Check if any RAC elements are missing from the provided dataframe
    missing_labels = [i for i in train_vars if i not in RAC_dataframe.columns]

    if len(missing_labels) > 0:
        # Try checking if there is anything in the column `RAC_column`. If so, deserialize it and re-run.
        if RAC_column in RAC_dataframe.columns:
            deserialized_RACs = pd.DataFrame.from_records(RAC_dataframe[RAC_column].values, index=RAC_dataframe.index.values)
            deserialized_RACs = deserialized_RACs.astype(float)
            RAC_dataframe = RAC_dataframe.join(deserialized_RACs)
            return perform_ANN_prediction(RAC_dataframe, predictor_name, RAC_column='RACs')
        else:
            raise ValueError('Please supply missing variables in your RAC dataframe: %s' % missing_labels)
    if 'alpha' in train_vars:
        if any(RAC_dataframe.alpha > 1):
            raise ValueError('Alpha is too large - should be between 0 and 1.')

    RAC_subset_for_ANN = RAC_dataframe.loc[:, train_vars].astype(float)
    normalized_input = data_normalize(RAC_subset_for_ANN, train_mean_x, train_var_x)
    ANN_prediction = my_ANN.predict(normalized_input, verbose=0)
    rescaled_output = data_rescale(ANN_prediction, train_mean_y, train_var_y)

    # Get latent vectors for training data and queried data
    train_x = pd.DataFrame(load_training_data(predictor_name), columns=train_vars).astype(float)
    get_outputs = K.function([my_ANN.layers[0].input, K.learning_phase()],
                             [my_ANN.layers[len(my_ANN.layers) - 2].output])
    normalized_train = data_normalize(train_x, train_mean_x, train_var_x)
    training_latent = get_outputs([normalized_train, 0])[0]
    query_latent = get_outputs([normalized_input, 0])[0]

    # Append all results to dataframe
    results_list = []
    for i in range(len(RAC_dataframe)):
        results_dict = {}
        min_latent_distance = min(np.linalg.norm(training_latent - query_latent[i][:], axis=1))
        results_dict['%s_latent_vector' % predictor_name] = query_latent[i]
        results_dict['%s_min_latent_distance' % predictor_name] = min_latent_distance
        output_value = rescaled_output[i]
        if len(output_value) == 1:  # squash array of length 1 to the value it contains
            output_value = output_value[0]
        results_dict['%s_prediction' % predictor_name] = output_value
        results_list.append(results_dict)
    results_df = pd.DataFrame(results_list, index=RAC_dataframe.index)
    RAC_dataframe_with_results = RAC_dataframe.join(results_df)
    return RAC_dataframe_with_results


def get_error_params(latent_distances, errors):
    '''
    Get the maximum-likelihood parameters for an error model N(a+b*(latent_distance)).
    Inputs: latent_distances (vector), errors (vector)
    Output: [a, b]
    '''
    def log_likelihood(params):
        a = params[0]
        b = params[1]
        return -np.nansum(scipy.stats.norm.logpdf(errors, loc=0, scale=a+latent_distances*b))
    results = scipy.optimize.minimize(log_likelihood, np.array([0.2, 0.01]), bounds=[(1e-9, None), (1e-9, None)])
    return results.x


def matrix_loader(path: str, rownames: bool = False) -> Union[Tuple[List[List[str]], List[str]], List[List[str]]]:
    # loads matrix with rowname option
    path_to_file = resource_files("molSimplify.python_nn").joinpath(path.strip("/"))
    if rownames:
        with open(path_to_file, "r") as f:
            csv_lines = list(csv.reader(f))
            row_names = [row[0] for row in csv_lines]
            mat = [row[1:] for row in csv_lines]
        return mat, row_names
    else:
        with open(path_to_file, 'r') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            mat = [a for a in lines]
        return mat


def get_key(predictor: str, suffix: Optional[str] = None) -> str:
    if suffix:
        if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
            key = f'geos/{predictor}_{suffix}'
        elif predictor in ['homo', 'gap']:
            key = f'homolumo/{predictor}_{suffix}'
        elif predictor in ['oxo', 'hat']:
            key = f'oxocatalysis/{predictor}_{suffix}'
        elif predictor in ['oxo20', 'homo_empty']:
            key = f'oxoandhomo/{predictor}_{suffix}'
        elif predictor in ['geo_static_clf', 'sc_static_clf']:
            key = f'{predictor}/{predictor}_{suffix}'
        else:
            key = f'{predictor}/{predictor}_{suffix}'
    else:
        if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
            key = 'geos/'
        elif predictor in ['homo', 'gap']:
            key = 'homolumo/'
        elif predictor in ['oxo', 'hat']:
            key = 'oxocatalysis/'
        elif predictor in ['oxo20', 'homo_empty']:
            key = 'oxoandhomo/'
        elif predictor in ['geo_static_clf', 'sc_static_clf']:
            key = f'{predictor}/{predictor}_{suffix}'
        else:
            key = predictor
    return key


def data_rescale(scaled_dat, train_mean, train_var, debug=False) -> np.ndarray:
    d = np.shape(train_mean)[0]
    if debug:
        print(f'unnormalizing with number of dimensions = {d}')
    dat = (np.multiply(scaled_dat.T, np.sqrt(train_var), ) + train_mean).T
    return (dat)


def data_normalize(data, train_mean, train_var, debug=False) -> np.ndarray:
    data = data.astype(float)  # Make sure the data is always in float form
    d = np.shape(train_mean)[0]
    # ## double check the variance in the training data
    delete_ind = list()

    if debug:
        print(f'normalizing with number of dimensions = {d}')
        print('shape of things in normalize:')
        print(f'data.shape {data.shape}')
        print(f'train_mean.shape {train_mean.shape}')
        print(f'train_var.shape {train_var.shape}')
    for idx, var in enumerate(np.squeeze(train_var)):
        if var < 1e-16:
            delete_ind.append(idx)
    if len(delete_ind) > 0:
        print(f'Note: There are {len(delete_ind)} features with a variance smaller than 1e-16.')
        print('Please double check your input data if this number is not what you expect...')
        data = np.delete(data, delete_ind, axis=1)
        train_mean = np.delete(train_mean, delete_ind, axis=0)
        train_var = np.delete(train_var, delete_ind, axis=0)

    scaled_dat = np.divide((data.T - train_mean), np.sqrt(train_var), ).T
    return scaled_dat


def load_normalization_data(name: str):
    train_mean_x = list()
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'rescaling_data/{name}_mean_x.csv')
    if os.path.isfile(path_to_file):
        with open(path_to_file, 'r') as f:
            for lines in f.readlines():
                train_mean_x.append([float(lines.strip().strip('[]'))])

        train_var_x = list()
        path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'rescaling_data/{name}_var_x.csv')
        with open(path_to_file, 'r') as f:
            for lines in f.readlines():
                train_var_x.append([float(lines.strip().strip('[]'))])

        train_mean_y = list()
        path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'rescaling_data/{name}_mean_y.csv')
        with open(path_to_file, 'r') as f:
            for lines in f.readlines():
                train_mean_y.append([float(lines.strip().strip('[]'))])
        train_var_y = list()
        path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'rescaling_data/{name}_var_y.csv')
        with open(path_to_file, 'r') as f:
            for lines in f.readlines():
                train_var_y.append([float(lines.strip().strip('[]'))])
    else:
        print('---Mean and Variance information do not exist. Calculate from training data...---')
        train_mean_x, train_mean_y, train_var_x, train_var_y = get_data_mean_std(predictor=name)

    return np.array(train_mean_x), np.array(train_mean_y), np.array(train_var_x), np.array(train_var_y)


def get_data_mean_std(predictor: str):
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = f'geos/{predictor}_bl_x'
    elif predictor in ['homo', 'gap']:
        key = f'homolumo/{predictor}_train_x'
    elif predictor in ['oxo', 'hat']:
        key = f'oxocatalysis/{predictor}_train_x'
    elif predictor in ['oxo20', 'homo_empty']:
        key = f'oxoandhomo/{predictor}_train_x'
    elif predictor == "split":
        key = f'{predictor}/{predictor}_x'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = f'{predictor}/{predictor}_train_x'
    else:
        key = f'{predictor}/{predictor}_x'
    path_to_feature_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    df_feature = pd.read_csv(path_to_feature_file)
    train_mean_x, train_var_x = list(), list()
    for col in df_feature:
        train_mean_x.append([np.mean(np.array(df_feature[col]))])
        train_var_x.append([np.var(np.array(df_feature[col]))])
    ### labels
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = f'geos/{predictor}_bl_y'
    elif predictor in ['homo', 'gap']:
        key = f'homolumo/{predictor}_train_y'
    elif predictor in ['oxo', 'hat']:
        key = f'oxocatalysis/{predictor}_train_y'
    elif predictor in ['oxo20', 'homo_empty']:
        key = f'oxoandhomo/{predictor}_train_y'
    elif predictor == "split":
        key = f'{predictor}/{predictor}_y'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = f'{predictor}/{predictor}_train_y'
    else:
        key = f'{predictor}/{predictor}_y'
    path_to_label_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    df_label = pd.read_csv(path_to_label_file)
    train_mean_y, train_var_y = list(), list()
    for col in df_label:
        train_mean_y.append([np.mean(np.array(df_label[col]))])
        train_var_y.append([np.var(np.array(df_label[col]))])
    return train_mean_x, train_mean_y, train_var_x, train_var_y


def load_ANN_variables(predictor: str, suffix: str = 'vars') -> List[str]:
    key = get_key(predictor, suffix)
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    names = []
    with open(path_to_file, 'r') as f:
        for lines in f.readlines():
            names.append(lines.strip())
    return names


def load_training_data(predictor: str) -> List[List[str]]:
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = f'geos/{predictor}_bl_x'
    elif predictor in ['homo', 'gap']:
        key = f'homolumo/{predictor}_train_x'
    elif predictor in ['oxo', 'hat']:
        key = f'oxocatalysis/{predictor}_train_x'
    elif predictor in ['oxo20', 'homo_empty']:
        key = f'oxoandhomo/{predictor}_train_x'
    elif predictor == "split":
        key = f'{predictor}/{predictor}_x'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = f'{predictor}/{predictor}_train_x'
    else:
        key = f'{predictor}/{predictor}_x'
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    with open(path_to_file, "r") as f:
        csv_lines = list(csv.reader(f))
        mat = [row for row in csv_lines[1:]]
    return mat


def load_latent_training_data(predictor):
    ##### CURRENTLY LATENT TRAINING DATA NOT AVAILABLE
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = f'geos/{predictor}_latent_bl_x'
    elif predictor in ['homo', 'gap']:
        key = f'homolumo/{predictor}_latent_train_x'
    elif predictor in ['oxo', 'hat']:
        key = f'oxocatalysis/{predictor}_latent_train_x'
    elif predictor in ['oxo20', 'homo_empty']:
        key = f'oxoandhomo/{predictor}_latent_train_x'
    elif predictor == "split":
        key = f'{predictor}/{predictor}_latent_x_41_OHE'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = f'{predictor}/{predictor}_latent_train_x'
    else:
        key = f'{predictor}/{predictor}_latent_x_OHE'
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    with open(path_to_file, "r") as f:
        csv_lines = list(csv.reader(f))
        mat = [row for row in csv_lines[1:]]
    return mat


def load_test_data(predictor):
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = f'geos/{predictor}_bl_x'  # Note, this test data is not available, will return train.
    elif predictor in ['homo', 'gap']:
        key = f'homolumo/{predictor}_test_x'
    elif predictor in ['oxo', 'hat']:
        key = f'oxocatalysis/{predictor}_test_x'
    elif predictor == "split":
        key = f'{predictor}/{predictor}_x'  # Note, this test data is not available, will return train
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = f'{predictor}/{predictor}_test_x'
    else:
        key = f'{predictor}/{predictor}_x'
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    with open(path_to_file, "r") as f:
        csv_lines = list(csv.reader(f))
        mat = [row for row in csv_lines[1:]]
    return mat


def load_training_labels(predictor: str) -> List[List[str]]:
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = f'geos/{predictor}_bl_y'
    elif predictor in ['homo', 'gap']:
        key = f'homolumo/{predictor}_train_y'
    elif predictor in ['oxo', 'hat']:
        key = f'oxocatalysis/{predictor}_train_y'
    elif predictor in ['oxo20', 'homo_empty']:
        key = f'oxoandhomo/{predictor}_train_y'
    elif predictor == "split":
        key = f'{predictor}/{predictor}_y'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = f'{predictor}/{predictor}_train_y'
    else:
        key = f'{predictor}/{predictor}_y'
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    with open(path_to_file, "r") as f:
        csv_lines = list(csv.reader(f))
        mat = [row for row in csv_lines[1:]]
    return mat


def load_test_labels(predictor: str) -> List[List[str]]:
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = f'geos/{predictor}_bl_y'
    elif predictor in ['homo', 'gap']:
        key = f'homolumo/{predictor}_test_y'
    elif predictor in ['oxo', 'hat']:
        key = f'oxocatalysis/{predictor}_test_y'
    elif predictor in ['oxo20', 'homo_empty']:
        key = f'oxoandhomo/{predictor}_test_y'
    elif predictor == "split":
        key = f'{predictor}/{predictor}_y'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = f'{predictor}/{predictor}_test_y'
    else:
        key = f'{predictor}/{predictor}_y'
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    with open(path_to_file, "rU") as f:
        csv_lines = list(csv.reader(f))
        mat = [row for row in csv_lines[1:]]
    return mat


def load_train_info(predictor: str, suffix: str = 'info') -> dict:
    key = get_key(predictor, suffix)
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.json')
    with open(path_to_file, 'r') as json_file:
        loaded_info_dict = json.loads(json_file.read())
    return loaded_info_dict


def load_keras_ann(predictor: str, suffix: str = 'model', compile: bool = False) -> tf.keras.Model:
    # this function loads the ANN for property
    # "predictor"
    # disable TF output text to reduce console spam

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    key = get_key(predictor, suffix)
    if "clf" not in predictor:
        path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.json')
        with open(path_to_file, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into  model
        path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.h5')
        loaded_model.load_weights(path_to_file)
    else:
        path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.h5')
        loaded_model = load_model(path_to_file)
    if compile:
        from tensorflow.keras.optimizers.legacy import Adam
        if predictor == 'homo':
            loaded_model.compile(loss="mse", optimizer=Adam(beta_2=1 - 0.0016204733101599046, beta_1=0.8718839135783554,
                                                            decay=7.770243145972892e-05, lr=0.0004961686075897741),
                                 metrics=['mse', 'mae', 'mape'])
        elif predictor == 'gap':
            loaded_model.compile(loss="mse", optimizer=Adam(beta_2=1 - 0.00010929248596488832, beta_1=0.8406735969305784,
                                                            decay=0.00011224350434148253, lr=0.0006759924688701965),
                                 metrics=['mse', 'mae', 'mape'])
        elif predictor in ['oxo', 'hat']:
            loaded_model.compile(loss="mse", optimizer=Adam(lr=0.0012838133056087084, beta_1=0.9811686522122317,
                                                            beta_2=0.8264616523572279, decay=0.0005114008091318582),
                                 metrics=['mse', 'mae', 'mape'])
        elif predictor == 'oxo20':
            loaded_model.compile(loss="mse", optimizer=Adam(lr=0.0012838133056087084, beta_1=0.9811686522122317,
                                                            beta_2=0.8264616523572279, decay=0.0005114008091318582),
                                 metrics=['mse', 'mae', 'mape'])
        elif predictor == 'homo_empty':
            loaded_model.compile(loss="mse", optimizer=Adam(lr=0.006677578283098809, beta_1=0.8556594887870226,
                                                            beta_2=0.9463468021275508, decay=0.0006621877134674607),
                                 metrics=['mse', 'mae', 'mape'])

        elif predictor in ['geo_static_clf', 'sc_static_clf']:
            loaded_model.compile(loss='binary_crossentropy',
                                 optimizer=Adam(lr=0.00005, beta_1=0.95, decay=0.0001, amsgrad=True),
                                 metrics=['accuracy'])
        else:
            loaded_model.compile(loss="mse", optimizer='adam',
                                 metrics=['mse', 'mae', 'mape'])
    return loaded_model



# --- sklearn loader + adapter ---
def _try_load_sklearn_classifier(predictor: str):
    """
    If an sklearn model for the classifier exists under sklearn_models/<predictor>/,
    load it with joblib and return an adapter that exposes predict/predict_proba.
    Otherwise return None and let Keras fallback take over.
    """
    import joblib

    here = Path(__file__).resolve()
    # Allow override via env var if you want (optional)
    base_override = os.environ.get("MOLSIMPLIFY_MODEL_DIR")  # e.g., /path/to/models
    candidates = []
    if base_override:
        candidates.append(Path(base_override) / "sklearn_models" / predictor)
    # common relative places
    candidates += [
        here.parent / "tf_nn" / predictor,               # molSimplify/python_nn/sklearn_models/geo_static_clf
        here.parent.parent / "tf_nn" / predictor,        # molSimplify/sklearn_models/geo_static_clf
        here.parent.parent.parent / "tf_nn" / predictor, # repo_root/sklearn_models/geo_static_clf
    ]

    filenames = ["model.joblib", "clf.joblib", "model.pkl", "clf.pkl"]
    for d in candidates:
        for fname in filenames:
            p = d / fname
            if p.exists():
                sk = joblib.load(p)

                class _SKAdapter:
                    def __init__(self, skobj, name):
                        self._sk = skobj
                        self.name = name
                    # mimic Keras .predict signature loosely
                    def predict(self, X, batch_size=None, verbose=None):
                        return self._sk.predict(X)
                    def predict_proba(self, X):
                        if hasattr(self._sk, "predict_proba"):
                            return self._sk.predict_proba(X)
                        raise AttributeError("This sklearn model has no predict_proba")
                    # handy for debugging/logs
                    def __repr__(self):
                        return f"<SKAdapter {self.name}: {self._sk.__class__.__name__}>"

                return _SKAdapter(sk, predictor)
    return None

def _fix_shape_for_model(model, X: np.ndarray) -> np.ndarray:
    """
    Ensure X matches model.input_shape:
    - If model expects (None, None, F) and X is (N, F) -> make (N, 1, F)
    - If model expects (None, F) and X is (N, 1, F)   -> squeeze to (N, F)
    Returns float32 array.
    """
    inp_shape = getattr(model, "input_shape", None)
    if isinstance(inp_shape, list):  # multi-input models → use first
        inp_shape = inp_shape[0]

    if inp_shape is not None:
        # Normalize negative/None dims away from batch dimension
        if len(inp_shape) == 3:
            # (batch, timesteps, features)
            if X.ndim == 2:
                X = np.expand_dims(X, axis=1)  # (N, 1, F)
        elif len(inp_shape) == 2:
            # (batch, features)
            if X.ndim == 3 and X.shape[1] == 1:
                X = X[:, 0, :]  # (N, F)
    else:
        # If model has no input_shape attr, fall back to “add a time axis if 2D”
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
    return X.astype(np.float32, copy=False)


def load_keras_ann2(predictor: str, suffix: str = "model", compile: bool = False):
    """
    Loads legacy Keras (JSON+H5) or monolithic H5 models from anywhere under molSimplify/tf_nn/.
    Searches recursively with glob so differing subdirectory layouts work automatically.
    """
    import os
    from pathlib import Path
    import glob
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # --- Keras imports ---
    try:
        try:
            from keras.saving.legacy.model_config import model_from_json as _legacy_model_from_json
        except Exception:
            _legacy_model_from_json = None
    except Exception:
        _legacy_model_from_json = None

    here = Path(__file__).resolve()
    tf_nn_root = here.parent.parent / "tf_nn"

    def _adam_compat(**kwargs):
        lr = kwargs.pop("lr", None)
        kwargs.pop("decay", None)
        if lr is not None:
            kwargs["learning_rate"] = lr
        return Adam(**kwargs)

    def _glob_find(basename):
        """
        Recursively search tf_nn/ for basename (case-sensitive).
        Returns the first match as a Path or None.
        """
        pattern = str(tf_nn_root / "**" / basename)
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return Path(hits[0])
        return None

    def _model_from_json_compat(model_json: str):
        """Handle keras 2.x JSONs."""
        if _legacy_model_from_json is not None:
            return _legacy_model_from_json(model_json)
        # fallback with explicit custom_objects
        try:
            from keras.models import model_from_json as _modern, Sequential
            from keras.layers import Dense, Dropout, BatchNormalization, Activation
            from keras.initializers import VarianceScaling, Zeros, Ones
            from keras.regularizers import L1L2
        except Exception:
            from tensorflow.keras.models import model_from_json as _modern, Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
            from tensorflow.keras.initializers import VarianceScaling, Zeros, Ones
            from tensorflow.keras.regularizers import L1L2
        custom_objects = {
            "Sequential": Sequential,
            "Dense": Dense,
            "Dropout": Dropout,
            "BatchNormalization": BatchNormalization,
            "Activation": Activation,
            "VarianceScaling": VarianceScaling,
            "Zeros": Zeros,
            "Ones": Ones,
            "L1L2": L1L2,
        }
        return _modern(model_json, custom_objects=custom_objects)

    # -------- actual loading --------
    if "clf" not in predictor:
        json_name = f"{predictor}_{suffix}.json"
        h5_name = f"{predictor}_{suffix}.h5"

        json_path = _glob_find(json_name)
        h5_path = _glob_find(h5_name)
        if json_path is None:
            raise FileNotFoundError(f"{json_name}")
        if h5_path is None:
            raise FileNotFoundError(f"{h5_name}")

        model_json = json_path.read_text()
        model = _model_from_json_compat(model_json)
        model.load_weights(h5_path)
    else:
        # classifier = single .h5 file
        h5_name = f"{predictor}_{suffix}.h5"
        h5_path = _glob_find(h5_name)
        if h5_path is None:
            raise FileNotFoundError(h5_name)
        model = load_model(h5_path, compile=False)

    # -------- optional compile --------
    if compile:
        if predictor == "homo":
            opt = _adam_compat(beta_2=1 - 0.0016204733101599046,
                               beta_1=0.8718839135783554,
                               lr=0.0004961686075897741)
            model.compile(loss="mse", optimizer=opt, metrics=["mse", "mae", "mape"])
        elif predictor == "gap":
            opt = _adam_compat(beta_2=1 - 0.00010929248596488832,
                               beta_1=0.8406735969305784,
                               lr=0.0006759924688701965)
            model.compile(loss="mse", optimizer=opt, metrics=["mse", "mae", "mape"])
        elif predictor in ["oxo", "hat", "oxo20"]:
            opt = _adam_compat(lr=0.0012838133056087084,
                               beta_1=0.9811686522122317,
                               beta_2=0.8264616523572279)
            model.compile(loss="mse", optimizer=opt, metrics=["mse", "mae", "mape"])
        elif predictor == "homo_empty":
            opt = _adam_compat(lr=0.006677578283098809,
                               beta_1=0.8556594887870226,
                               beta_2=0.9463468021275508)
            model.compile(loss="mse", optimizer=opt, metrics=["mse", "mae", "mape"])
        elif predictor in ["geo_static_clf", "sc_static_clf"]:
            opt = _adam_compat(lr=5e-5, beta_1=0.95, amsgrad=True)
            model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        else:
            model.compile(loss="mse", optimizer=Adam(), metrics=["mse", "mae", "mape"])

    return model



def tf_ANN_excitation_prepare(predictor: str, descriptors: List[float], descriptor_names: List[str]) -> np.ndarray:
    ## this function reforms the provided list of descriptors and their
    ## names to match the expectations of the target ANN model.
    ## it does NOT perform standardization

    ## get variable names
    target_names = load_ANN_variables(predictor)
    if len(target_names) > len(descriptors):
        print(f'Error: preparing features for {predictor}, received {len(descriptors)} descriptors')
        print(f'model requires {len(target_names)} descriptors, attempting match')
    excitation = []
    for var_name in target_names:
        try:
            excitation.append(descriptors[descriptor_names.index(var_name)])
        except ValueError:
            print(f'looking for {var_name}')
            print(f'Error! variable {var_name} not found!')
            break
    output = np.array(excitation)
    output = np.reshape(output, (1, len(target_names)))
    return output


def ANN_supervisor(predictor: str,
                   descriptors: List[float],
                   descriptor_names: List[str],
                   debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if debug:
        print(f'ANN activated for {predictor}')

    ## form the excitation in the correct order/variables
    excitation = tf_ANN_excitation_prepare(predictor, descriptors, descriptor_names)
    if debug:
        print(f'excitation is {excitation.shape}')
        print('fetching non-dimensionalization data... ')
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)
    if debug:
        print('rescaling input excitation...')

    excitation = data_normalize(excitation, train_mean_x, train_var_x, debug=debug)

    ## fetch ANN
    try:
        loaded_model = load_keras_ann(predictor)
        result = data_rescale(loaded_model.predict(excitation, verbose=0), train_mean_y, train_var_y, debug=debug)
    except:
        loaded_model = load_keras_ann2(predictor)
        excitation = _fix_shape_for_model(loaded_model, excitation)
        result = data_rescale(loaded_model.predict(excitation, verbose=0), train_mean_y, train_var_y, debug=debug)

    if "clf" not in predictor:
        if debug:
            print(f'LOADED MODEL HAS {len(loaded_model.layers)} layers, so latent space measure will be from first {len(loaded_model.layers) - 1} layers')
        if version.parse(tf.__version__) >= version.parse('2.0.0'):
            latent_space_vector = get_layer_outputs(loaded_model, len(loaded_model.layers) - 2,
                                                    excitation, training_flag=False)
        else:
            get_outputs = K.function([loaded_model.layers[0].input, K.learning_phase()],
                                     [loaded_model.layers[len(loaded_model.layers) - 2].output])
            latent_space_vector = get_outputs([excitation, 0])  # Using test phase.
        if debug:
            print('calling ANN model...')
    else:
        latent_space_vector = find_clf_lse(predictor, excitation, loaded_model=loaded_model, ensemble=False,
                                           modelname=None, debug=debug)
    return result, latent_space_vector


def find_true_min_eu_dist(predictor: str,
                          descriptors: List[float],
                          descriptor_names: List[str],
                          debug: bool = False) -> float:
    # returns scaled euclidean distance to nearest trainning
    # vector in desciptor space
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)

    ## form the excitation in the corrrect order/variables
    excitation = tf_ANN_excitation_prepare(predictor, descriptors, descriptor_names)
    excitation = excitation.astype(float)  # ensure that the excitation is a float, and not strings
    scaled_excitation = data_normalize(excitation, train_mean_x, train_var_x,  debug=debug)  # normalize the excitation
    ## getting train matrix info
    mat = load_training_data(predictor)
    train_mat = np.array(mat, dtype='float64')
    ## loop over rows
    min_dist = np.inf
    min_ind = 0
    for i, rows in enumerate(train_mat):
        scaled_row = np.squeeze(
            data_normalize(rows, train_mean_x.T, train_var_x.T, debug=debug))  # Normalizing the row before finding the distance
        this_dist = float(np.linalg.norm(np.subtract(scaled_row, np.array(scaled_excitation))))  # Cast to float for mypy typing
        if this_dist < min_dist:
            min_dist = this_dist
            min_ind = i

    if debug:
        print(f'min dist EU is {min_dist}')
    folder_dict = {'homo': 'homolumo', 'gap': 'homolumo',
                   'oxo': 'oxocatalysis', 'hat': 'oxocatalysis',
                   'oxo20': 'oxoandhomo', 'homo_empty': 'oxoandhomo'}
    if predictor in folder_dict:
        key = f'{folder_dict[predictor]}/{predictor}_train_names'
        path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
        with open(path_to_file, "r") as f:
            csv_lines = list(csv.reader(f))
            print(f'Closest Euc Dist Structure: {str(csv_lines[min_ind]).strip("[]")} for predictor {predictor}')
    # need to get normalized distances

    ########################################################################################
    # Changed by Aditya on 08/13/2018. Previously, nearest neighbor was being found in the #
    # unnormalized space, and then that was normalized. This was resulting in bad nearest  #
    # neighbor candidate structures. Now routine normalizes before finding the distance.   #
    ########################################################################################

    return (min_dist)


def find_ANN_10_NN_normalized_latent_dist(predictor, latent_space_vector, debug=False):
    # returns scaled euclidean distance to nearest trainning
    # vector in desciptor space

    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)

    ## getting train matrix info
    mat = load_training_data(predictor)
    train_mat = np.array(mat, dtype='float64')

    loaded_model = load_keras_ann(predictor)
    if debug:
        print('measuring latent distances:')
        print(f'loaded model has {len(loaded_model.layers)} layers, so latent space measure will be from first {len(loaded_model.layers) - 1} layers')
    norm_train_mat = []
    for i, row in enumerate(train_mat):
        row = np.array(row)
        scaled_excitation = data_normalize(row, train_mean_x.T, train_var_x.T)
        norm_train_mat.append(scaled_excitation)
    norm_train_mat = np.squeeze(np.array(norm_train_mat))
    loaded_model = load_keras_ann(predictor)
    if version.parse(tf.__version__) >= version.parse('2.0.0'):
        latent_space_train = get_layer_outputs(loaded_model, len(loaded_model.layers) - 2,
                                               norm_train_mat, training_flag=False)
        latent_space_train = np.squeeze(np.array(latent_space_train))
    else:
        get_outputs = K.function([loaded_model.layers[0].input, K.learning_phase()],
                                 [loaded_model.layers[len(loaded_model.layers) - 2].output])
        latent_space_train = np.squeeze(np.array(get_outputs([norm_train_mat, 0])))
    dist_array = np.linalg.norm(np.subtract(np.squeeze(latent_space_train), np.squeeze(latent_space_vector)), axis=1)
    from scipy.spatial import distance_matrix
    train_dist_array = distance_matrix(latent_space_train, latent_space_train)
    nearest_10_NN_train = []
    for j, train_row in enumerate(train_dist_array):
        nearest_10_NN_train.append(np.sort(np.squeeze(train_row))[0:10])
    nearest_10_NN_train = np.array(nearest_10_NN_train)
    avg_traintrain = np.mean(nearest_10_NN_train)
    sorted_dist = np.sort(np.squeeze(dist_array))
    avg_10_NN_dist = np.mean(sorted_dist[0:10])
    norm_avg_10_NN_dist = avg_10_NN_dist/avg_traintrain
    return norm_avg_10_NN_dist, avg_10_NN_dist, avg_traintrain


def find_ANN_latent_dist(predictor, latent_space_vector, debug=False):
    # returns scaled euclidean distance to nearest trainning
    # vector in desciptor space
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)

    ## getting train matrix info
    mat = load_training_data(predictor)
    train_mat = np.array(mat, dtype='float64')
    ## loop over rows
    min_dist = 100000000
    min_ind = 0

    try:
        loaded_model = load_keras_ann(predictor)
    except:
        loaded_model = load_keras_ann2(predictor)


    if debug:
        print('measuring latent distances:')
        print(f'loaded model has {len(loaded_model.layers)} layers, so latent space measure will be from first {len(loaded_model.layers) - 1} layers')
    if version.parse(tf.__version__) >= version.parse('2.0.0'):
        get_outputs = None
    else:
        get_outputs = K.function([loaded_model.layers[0].input, K.learning_phase()],
                                 [loaded_model.layers[len(loaded_model.layers) - 2].output])

    for i, rows in enumerate(train_mat):
        scaled_row = np.squeeze(
            data_normalize(rows, train_mean_x.T, train_var_x.T, debug=debug))  # Normalizing the row before finding the distance
        if version.parse(tf.__version__) >= version.parse('2.0.0'):
            try:
                latent_train_row = get_layer_outputs(loaded_model, len(loaded_model.layers) - 2,
                                                     [np.array([scaled_row])], training_flag=False)
            except:
                latent_train_row = get_layer_outputs2(loaded_model, len(loaded_model.layers) - 2,
                                                     [np.array([scaled_row])], training_flag=False)
        else:
            latent_train_row = get_outputs([np.array([scaled_row]), 0])
        this_dist = np.linalg.norm(np.subtract(np.squeeze(latent_train_row), np.squeeze(latent_space_vector)))
        if this_dist < min_dist:
            min_dist = this_dist
            min_ind = i

    # flatten min row
    if debug:
        print(f'min dist is {min_dist} at {min_ind}')
    folder_dict = {'homo': 'homolumo', 'gap': 'homolumo',
                   'oxo': 'oxocatalysis', 'hat': 'oxocatalysis',
                   'oxo20': 'oxoandhomo', 'homo_empty': 'oxoandhomo'}
    if predictor in folder_dict:
        key = f'{folder_dict[predictor]}/{predictor}_train_names'
        path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
        with open(path_to_file, "r") as f:
            csv_lines = list(csv.reader(f))
            print(f'Closest Latent Dist Structure: {csv_lines[min_ind]} for predictor {predictor}')
    return (min_dist)


def find_clf_lse(predictor: str,
                 excitation,
                 loaded_model,
                 ensemble: bool = False,
                 modelname: Optional[str] = None,
                 debug: bool = False) -> np.ndarray:
    if modelname is None:
        modelname = "spectro"
    if predictor == "geo_static_clf":
        avrg_latent_dist = 33.21736244173539
    elif predictor == "sc_static_clf":
        avrg_latent_dist = 38.276809428032685
    else:
        print("Unknown model type")
        return np.zeros_like(excitation)
    key = get_key(predictor, suffix='')
    base_path = resource_files("molSimplify.tf_nn").joinpath(key)
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)
    labels_train = np.array(load_training_labels(predictor), dtype='int')
    fmat_train = np.array(load_training_data(predictor), dtype='float64')
    fmat_train = data_normalize(fmat_train, train_mean_x, train_var_x,  debug=debug)
    fmat_train = np.array(fmat_train)

    if ensemble:
        print("Using ensemble averaged LSE.")
        base_path = f'{base_path}ensemble_{modelname}/'
        model_list = sorted(glob.glob(base_path + '/*.h5'))
        if len(model_list) != 10:
            print(key)
            print(base_path)
            print(model_list)
            print(f"Error: LSE cannot be calculated with modelname {modelname}--The number of models is wrong.")
            return np.zeros_like(excitation)
        fmat_train_split = np.array_split(fmat_train, 10, axis=0)
        labels_train_split = np.array_split(labels_train, 10, axis=0)
        entropies_list = []
        for model in model_list:
            print(model)
            loaded_model = load_model(model)
            model_idx = int(model.split("/")[-1].split(".")[0].split("_")[-1])
            _fmat_train = array_stack(fmat_train_split, model_idx)
            _labels_train = array_stack(labels_train_split, model_idx)
            train_latent = get_layer_outputs(loaded_model, -4, _fmat_train, training_flag=False)
            test_latent = get_layer_outputs(loaded_model, -4, excitation, training_flag=False)
            nn_latent_dist_train, _, __ = dist_neighbor(train_latent, train_latent, _labels_train,
                                                        l=5, dist_ref=1)
            avrg_latent_dist = np.mean(nn_latent_dist_train)
            nn_latent_dist_test, nn_dists, nn_labels = dist_neighbor(test_latent, train_latent, _labels_train,
                                                                     l=5, dist_ref=avrg_latent_dist)
            entropies = get_entropy(nn_dists, nn_labels)
            entropies_list.append(entropies)
        lse = np.mean(np.array(entropies_list), axis=0)
    else:
        train_latent = get_layer_outputs(loaded_model, -4, fmat_train, training_flag=False)
        test_latent = get_layer_outputs(loaded_model, -4, excitation, training_flag=False)
        nn_latent_dist_test, nn_dists, nn_labels = dist_neighbor(test_latent, train_latent, labels_train,
                                                                 l=5, dist_ref=avrg_latent_dist)
        lse = get_entropy(nn_dists, nn_labels)
    return lse


def save_model(model: tf.keras.Model, predictor: str,
               num: Optional[int] = None, suffix: Optional[str] = None):
    key = get_key(predictor, suffix)
    base_path = resource_files("molSimplify.tf_nn").joinpath(key)
    base_path = base_path + 'ensemble_models'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if num is not None:
        name = '%s/%s_%d' % (base_path, predictor, num)
    else:
        name = '%s/%s' % (base_path, predictor)
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s.json" % name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{name}.h5")
    print("Saved model !%s! to disk" % name.split('/')[-1])


def initialize_model_weights(model: tf.keras.Model) -> tf.keras.Model:
    session = K.get_session()
    for layer in model.layers:
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
    return model
