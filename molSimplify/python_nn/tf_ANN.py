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
from tensorflow.keras.models import model_from_json, load_model
from importlib_resources import files as resource_files
from packaging import version
import tensorflow as tf

from molSimplify.python_nn.clf_analysis_tool import array_stack, get_layer_outputs, dist_neighbor, get_entropy


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
            deserialized_RACs = pd.DataFrame.from_records(RAC_dataframe[RAC_column].values, index=RAC_dataframe.index)
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
            key = 'geos/' + predictor + '_%s' % suffix
        elif predictor in ['homo', 'gap']:
            key = 'homolumo/' + predictor + '_%s' % suffix
        elif predictor in ['oxo', 'hat']:
            key = 'oxocatalysis/' + predictor + '_%s' % suffix
        elif predictor in ['oxo20', 'homo_empty']:
            key = 'oxoandhomo/' + predictor + '_%s' % suffix
        elif predictor in ['geo_static_clf', 'sc_static_clf']:
            key = predictor + '/' + predictor + '_%s' % suffix
        else:
            key = predictor + '/' + predictor + '_%s' % suffix
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
            key = predictor + '/' + predictor + '_%s' % suffix
        else:
            key = predictor
    return key


def data_rescale(scaled_dat, train_mean, train_var, debug=False) -> np.ndarray:
    d = np.shape(train_mean)[0]
    if debug:
        print(('unnormalizing with number of dimensions = ' + str(d)))
    dat = (np.multiply(scaled_dat.T, np.sqrt(train_var), ) + train_mean).T
    return (dat)


def data_normalize(data, train_mean, train_var, debug=False) -> np.ndarray:
    data = data.astype(float)  # Make sure the data is always in float form
    d = np.shape(train_mean)[0]
    # ## double check the variance in the training data
    delete_ind = list()

    if debug:
        print(('normalizing with number of dimensions = ' + str(d)))
        print('shape of things in normalize:')
        print(('data.shape ' + str(data.shape)))
        print(('train_mean.shape ' + str(train_mean.shape)))
        print(('train_mean.shape ' + str(train_var.shape)))
    for idx, var in enumerate(np.squeeze(train_var)):
        if var < 1e-16:
            delete_ind.append(idx)
    if len(delete_ind) > 0:
        print(('Note: There are %d features with a variance smaller than 1e-16.' % len(delete_ind)))
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
        key = 'geos/' + predictor + '_bl_x'
    elif predictor in ['homo', 'gap']:
        key = 'homolumo/' + predictor + '_train_x'
    elif predictor in ['oxo', 'hat']:
        key = 'oxocatalysis/' + predictor + '_train_x'
    elif predictor in ['oxo20', 'homo_empty']:
        key = 'oxoandhomo/' + predictor + '_train_x'
    elif predictor == "split":
        key = predictor + '/' + predictor + '_x'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = f'{predictor}/{predictor}_train_x'
    else:
        key = predictor + '/' + predictor + '_x'
    path_to_feature_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    df_feature = pd.read_csv(path_to_feature_file)
    train_mean_x, train_var_x = list(), list()
    for col in df_feature:
        train_mean_x.append([np.mean(np.array(df_feature[col]))])
        train_var_x.append([np.var(np.array(df_feature[col]))])
    ### labels
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = 'geos/' + predictor + '_bl_y'
    elif predictor in ['homo', 'gap']:
        key = 'homolumo/' + predictor + '_train_y'
    elif predictor in ['oxo', 'hat']:
        key = 'oxocatalysis/' + predictor + '_train_y'
    elif predictor in ['oxo20', 'homo_empty']:
        key = 'oxoandhomo/' + predictor + '_train_y'
    elif predictor == "split":
        key = predictor + '/' + predictor + '_y'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = f'{predictor}/{predictor}_train_y'
    else:
        key = predictor + '/' + predictor + '_y'
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
        key = 'geos/' + predictor + '_bl_x'
    elif predictor in ['homo', 'gap']:
        key = 'homolumo/' + predictor + '_train_x'
    elif predictor in ['oxo', 'hat']:
        key = 'oxocatalysis/' + predictor + '_train_x'
    elif predictor in ['oxo20', 'homo_empty']:
        key = 'oxoandhomo/' + predictor + '_train_x'
    elif predictor == "split":
        key = predictor + '/' + predictor + '_x'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = predictor + '/' + predictor + '_train_x'
    else:
        key = predictor + '/' + predictor + '_x'
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    with open(path_to_file, "r") as f:
        csv_lines = list(csv.reader(f))
        # row_names = [row[0] for row in csv_lines]
        mat = [row for row in csv_lines[1:]]
    return mat


def load_latent_training_data(predictor):
    ##### CURRENTLY LATENT TRAINING DATA NOT AVAIL
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = 'geos/' + predictor + '_latent_bl_x'
    elif predictor in ['homo', 'gap']:
        key = 'homolumo/' + predictor + '_latent_train_x'
    elif predictor in ['oxo', 'hat']:
        key = 'oxocatalysis/' + predictor + '_latent_train_x'
    elif predictor in ['oxo20', 'homo_empty']:
        key = 'oxoandhomo/' + predictor + '_latent_train_x'
    elif predictor == "split":
        key = predictor + '/' + predictor + '_latent_x_41_OHE'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = predictor + '/' + predictor + '_latent_train_x'
    else:
        key = predictor + '/' + predictor + '_latent_x_OHE'
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    with open(path_to_file, "r") as f:
        csv_lines = list(csv.reader(f))
        # row_names = [row[0] for row in csv_lines]
        mat = [row for row in csv_lines[1:]]
    return mat


def load_test_data(predictor):
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = 'geos/' + predictor + '_bl_x'  # Note, this test data is not available, will return train.
    elif predictor in ['homo', 'gap']:
        key = 'homolumo/' + predictor + '_test_x'
    elif predictor in ['oxo', 'hat']:
        key = 'oxocatalysis/' + predictor + '_test_x'
    elif predictor == "split":
        key = predictor + '/' + predictor + '_x'  # Note, this test data is not available, will return train
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = predictor + '/' + predictor + '_test_x'
    else:
        key = predictor + '/' + predictor + '_x'
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    with open(path_to_file, "r") as f:
        csv_lines = list(csv.reader(f))
        # row_names = [row[0] for row in csv_lines]
        mat = [row for row in csv_lines[1:]]
    return mat


def load_training_labels(predictor: str) -> List[List[str]]:
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = 'geos/' + predictor + '_bl_y'
    elif predictor in ['homo', 'gap']:
        key = 'homolumo/' + predictor + '_train_y'
    elif predictor in ['oxo', 'hat']:
        key = 'oxocatalysis/' + predictor + '_train_y'
    elif predictor in ['oxo20', 'homo_empty']:
        key = 'oxoandhomo/' + predictor + '_train_y'
    elif predictor == "split":
        key = predictor + '/' + predictor + '_y'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = predictor + '/' + predictor + '_train_y'
    else:
        key = predictor + '/' + predictor + '_y'
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    with open(path_to_file, "r") as f:
        csv_lines = list(csv.reader(f))
        # row_names = [row[0] for row in csv_lines]
        mat = [row for row in csv_lines[1:]]
    return mat


def load_test_labels(predictor: str) -> List[List[str]]:
    if predictor in ['ls_ii', 'hs_ii', 'ls_iii', 'hs_iii']:
        key = 'geos/' + predictor + '_bl_y'
    elif predictor in ['homo', 'gap']:
        key = 'homolumo/' + predictor + '_test_y'
    elif predictor in ['oxo', 'hat']:
        key = 'oxocatalysis/' + predictor + '_test_y'
    elif predictor in ['oxo20', 'homo_empty']:
        key = 'oxoandhomo/' + predictor + '_test_y'
    elif predictor == "split":
        key = predictor + '/' + predictor + '_y'
    elif predictor in ['geo_static_clf', 'sc_static_clf']:
        key = predictor + '/' + predictor + '_test_y'
    else:
        key = predictor + '/' + predictor + '_y'
    path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
    with open(path_to_file, "rU") as f:
        csv_lines = list(csv.reader(f))
        # row_names = [row[0] for row in csv_lines]
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
    # "predcitor"
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
            # loaded_model.compile(loss="mse", optimizer=Adam(beta_2=0.9637165412871632, beta_1=0.7560951483268549,
            #                                                 decay=0.0006651401379502965, lr=0.0007727366541920176),
            #                      metrics=['mse', 'mae', 'mape']) #decomissioned on 06/20/2019 by Aditya. Using hyperparams from oxo20.
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
    # print("Keras/tf model loaded for " + str(predictor) + " from disk")
    return loaded_model


def tf_ANN_excitation_prepare(predictor: str, descriptors: List[float], descriptor_names: List[str]) -> np.ndarray:
    ## this function reforms the provided list of descriptors and their
    ## names to match the expectations of the target ANN model.
    ## it does NOT perfrom standardization

    ## get variable names
    target_names = load_ANN_variables(predictor)
    if len(target_names) > len(descriptors):
        print('Error: preparing features for ' + str(predictor) + ', recieved '
              + str(len(descriptors)) + ' descriptors')
        print(('model requires ' + str(len(target_names)) + ' descriptors, attempting match'))
    excitation = []
    for var_name in target_names:
        try:
            excitation.append(descriptors[descriptor_names.index(var_name)])
        except ValueError:
            print(('looking for  ' + str(var_name)))
            print(('Error! variable  ' + str(var_name) + ' not found!'))
            break
    output = np.array(excitation)
    output = np.reshape(output, (1, len(target_names)))
    return output


def ANN_supervisor(predictor: str,
                   descriptors: List[float],
                   descriptor_names: List[str],
                   debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if debug:
        print(('ANN activated for ' + str(predictor)))

    ## form the excitation in the corrrect order/variables
    excitation = tf_ANN_excitation_prepare(predictor, descriptors, descriptor_names)
    if debug:
        print(('excitation is ' + str(excitation.shape)))
        print('fetching non-dimensionalization data... ')
    # sardines
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)
    if debug:
        print('rescaling input excitation...')

    excitation = data_normalize(excitation, train_mean_x, train_var_x, debug=debug)

    ## fetch ANN
    loaded_model = load_keras_ann(predictor)
    result = data_rescale(loaded_model.predict(excitation, verbose=0), train_mean_y, train_var_y, debug=debug)
    if "clf" not in predictor:
        if debug:
            print(('LOADED MODEL HAS ' + str(
                len(loaded_model.layers)) + ' layers, so latent space measure will be from first ' + str(
                len(loaded_model.layers) - 1) + ' layers'))
        if not version.parse(tf.__version__) >= version.parse('2.0.0'):
            get_outputs = K.function([loaded_model.layers[0].input, K.learning_phase()],
                                     [loaded_model.layers[len(loaded_model.layers) - 2].output])
            latent_space_vector = get_outputs([excitation, 0])  # Using test phase.
        else:
            latent_space_vector = get_layer_outputs(loaded_model, len(loaded_model.layers) - 2,
                                                    excitation, training_flag=False)
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
            # best_row = rownames[i]
            # min_row = rows

    if debug:
        print(('min dist EU is ' + str(min_dist)))
    folder_dict = {'homo': 'homolumo', 'gap': 'homolumo',
                   'oxo': 'oxocatalysis', 'hat': 'oxocatalysis',
                   'oxo20': 'oxoandhomo', 'homo_empty': 'oxoandhomo'}
    if predictor in folder_dict:
        key = f'{folder_dict[predictor]}/{predictor}_train_names'
        path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
        with open(path_to_file, "r") as f:
            csv_lines = list(csv.reader(f))
            print(('Closest Euc Dist Structure:  ' + str(csv_lines[min_ind]).strip('[]') + ' for predictor ' + str(
                predictor)))
    # need to get normalized distances

    ########################################################################################
    # Changed by Aditya on 08/13/2018. Previously, nearest neighbor was being found in the #
    # unnormalized space, and then that was normalized. This was resulting in bad nearest  #
    # neighbor candidate structures. Now routine normalizes before finding the distance.   #
    ########################################################################################

    # train_mean_x,train_mean_y,train_var_x,train_var_y = load_normalization_data(predictor)

    # flatten min row
    # min_row = np.reshape(min_row, excitation.shape)
    # scaled_excitation = data_normalize(excitation,train_mean_x,train_var_x)
    # scaled_row = data_normalize(min_row,train_mean_x,train_var_x)
    # min_dist = np.linalg.norm(np.subtract(scaled_row,(scaled_excitation)))
    return (min_dist)


def find_ANN_10_NN_normalized_latent_dist(predictor, latent_space_vector, debug=False):
    # returns scaled euclidean distance to nearest trainning
    # vector in desciptor space

    # average_train_train_10NN = {'homo_empty': 0.43517572, 'oxo20': 0.068675719}
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)

    ## getting train matrix info
    mat = load_training_data(predictor)
    train_mat = np.array(mat, dtype='float64')

    loaded_model = load_keras_ann(predictor)
    if debug:
        print('measuring latent distances:')
        print(('loaded model has  ' + str(
            len(loaded_model.layers)) + ' layers, so latent space measure will be from first ' + str(
            len(loaded_model.layers) - 1) + ' layers'))
    norm_train_mat = []
    for i, row in enumerate(train_mat):
        row = np.array(row)
        scaled_excitation = data_normalize(row, train_mean_x.T, train_var_x.T)
        norm_train_mat.append(scaled_excitation)
    norm_train_mat = np.squeeze(np.array(norm_train_mat))
    loaded_model = load_keras_ann(predictor)
    if not version.parse(tf.__version__) >= version.parse('2.0.0'):
        get_outputs = K.function([loaded_model.layers[0].input, K.learning_phase()],
                                 [loaded_model.layers[len(loaded_model.layers) - 2].output])
        latent_space_train = np.squeeze(np.array(get_outputs([norm_train_mat, 0])))
    else:
        latent_space_train = get_layer_outputs(loaded_model, len(loaded_model.layers) - 2,
                                               norm_train_mat, training_flag=False)
        latent_space_train = np.squeeze(np.array(latent_space_train))
    dist_array = np.linalg.norm(np.subtract(np.squeeze(latent_space_train), np.squeeze(latent_space_vector)), axis=1)
    # train_dist_array =  np.linalg.norm(np.subtract(np.squeeze(latent_space_train), np.squeeze(latent_space_train)),axis=1)
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

    loaded_model = load_keras_ann(predictor)

    if debug:
        print('measuring latent distances:')
        print(('loaded model has  ' + str(
            len(loaded_model.layers)) + ' layers, so latent space measure will be from first ' + str(
            len(loaded_model.layers) - 1) + ' layers'))
    if not version.parse(tf.__version__) >= version.parse('2.0.0'):
        get_outputs = K.function([loaded_model.layers[0].input, K.learning_phase()],
                                 [loaded_model.layers[len(loaded_model.layers) - 2].output])
    for i, rows in enumerate(train_mat):
        scaled_row = np.squeeze(
            data_normalize(rows, train_mean_x.T, train_var_x.T, debug=debug))  # Normalizing the row before finding the distance
        if not version.parse(tf.__version__) >= version.parse('2.0.0'):
            latent_train_row = get_outputs([np.array([scaled_row]), 0])
        else:
            latent_train_row = get_layer_outputs(loaded_model, len(loaded_model.layers) - 2,
                                                 [np.array([scaled_row])], training_flag=False)
        this_dist = np.linalg.norm(np.subtract(np.squeeze(latent_train_row), np.squeeze(latent_space_vector)))
        if this_dist < min_dist:
            min_dist = this_dist
            min_ind = i

    # flatten min row
    if debug:
        print(('min dist is ' + str(min_dist) + ' at  ' + str(min_ind)))
    folder_dict = {'homo': 'homolumo', 'gap': 'homolumo',
                   'oxo': 'oxocatalysis', 'hat': 'oxocatalysis',
                   'oxo20': 'oxoandhomo', 'homo_empty': 'oxoandhomo'}
    if predictor in folder_dict:
        key = f'{folder_dict[predictor]}/{predictor}_train_names'
        path_to_file = resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv')
        with open(path_to_file, "r") as f:
            csv_lines = list(csv.reader(f))
            print(('Closest Latent Dist Structure: ' + str(csv_lines[min_ind]) + ' for predictor ' + str(predictor)))
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
    if not ensemble:
        # model = base_path + 'model.h5'
        # loaded_model = load_model(model)
        train_latent = get_layer_outputs(loaded_model, -4, fmat_train, training_flag=False)
        test_latent = get_layer_outputs(loaded_model, -4, excitation, training_flag=False)
        nn_latent_dist_test, nn_dists, nn_labels = dist_neighbor(test_latent, train_latent, labels_train,
                                                                 l=5, dist_ref=avrg_latent_dist)
        lse = get_entropy(nn_dists, nn_labels)
    else:
        print("Using ensemble averaged LSE.")
        base_path = base_path + 'ensemble_%s/' % modelname
        model_list = sorted(glob.glob(base_path + '/*.h5'))
        if len(model_list) != 10:
            print(key)
            print(base_path)
            print(model_list)
            print(("Error: LSE cannot be calculated with modelname %s--The number of models is wrong." % modelname))
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
    model.save_weights("%s.h5" % name)
    print(("Saved model !%s! to disk" % name.split('/')[-1]))


def initialize_model_weights(model: tf.keras.Model) -> tf.keras.Model:
    session = K.get_session()
    for layer in model.layers:
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
                # print('reinitializing layer {}.{}'.format(layer.name, v))
    return model
