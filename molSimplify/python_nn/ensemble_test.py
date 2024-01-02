import glob
import math

import numpy as np
import scipy as sp
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K

from importlib_resources import files as resource_files
from sklearn.utils import shuffle
from molSimplify.python_nn.clf_analysis_tool import (dist_neighbor,
                                                     get_entropy)
from molSimplify.python_nn.tf_ANN import (data_normalize,
                                          data_rescale,
                                          get_key,
                                          load_keras_ann,
                                          load_normalization_data,
                                          load_test_data,
                                          load_test_labels,
                                          load_train_info,
                                          load_training_data,
                                          load_training_labels,
                                          save_model,
                                          tf_ANN_excitation_prepare)


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
    return model


def array_stack(array, _idx):
    for ii in range(len(array)):
        if not ii == _idx:
            if 'out_arr' not in dir():
                out_arr = array[ii]
            else:
                out_arr = np.concatenate((out_arr, array[ii]))
    return out_arr


def mc_dropout_logp(tau, err):
    T = err.shape[0]
    logp = sp.misc.logsumexp(-0.5 * tau * err)
    logp -= np.log(T)
    logp -= 0.5 * np.log(np.power(tau, -1))
    logp -= 0.5 * np.log(2 * math.pi)
    return (logp)


def ensemble_maker_inner(train_mat, labels, model_gen_function, info_dict,
                         num=10):
    ## contains core functions to make ensemble models
    ## from training data and labels
    ## model_gen_function is a functiont that takes NO arguments and returns a keras model
    ## info_dict is a dictionary of training info
    train_mat, labels = shuffle(train_mat, labels)
    train_mat = np.array_split(train_mat, num, axis=0)
    labels = np.array_split(labels, num, axis=0)
    earlystop = EarlyStopping(monitor=info_dict['monitor'],
                              min_delta=info_dict['min_delta'],
                              patience=info_dict['patience'],
                              verbose=0,
                              mode='auto')
    callbacks_list = [earlystop]
    model_list = []
    for ii in range(num):
        train_feature = array_stack(train_mat, ii)
        train_labels = array_stack(labels, ii)
        loaded_model = model_gen_function()  # note the call to gen new model
        current_model = reset_weights(loaded_model)
        current_model.fit(train_feature, train_labels,
                          epochs=info_dict['epochs'], verbose=0,
                          batch_size=info_dict['batch_size'],
                          callbacks=callbacks_list)
        model_list.append(current_model)
    return(model_list)


def ensemble_maker(predictor, num=10):
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)
    mat = load_training_data(predictor)
    mat = np.array(mat, dtype='float64')
    train_mat = data_normalize(mat, train_mean_x, train_var_x)
    labels = load_training_labels(predictor)
    if "clf" not in predictor:
        labels = np.array(labels, dtype='float64')
        labels = data_normalize(labels, train_mean_y, train_var_y)
    info_dict = load_train_info(predictor)
    model_list = ensemble_maker_inner(train_mat=train_mat,
                                      labels=labels,
                                      model_gen_function=lambda: load_keras_ann(predictor),
                                      info_dict=info_dict, num=num)
    for ii, current_model in enumerate(model_list):
        save_model(current_model, predictor, ii)


def ensemble_uq(predictor, descriptors=False, descriptor_names=False, suffix=False):
    key = get_key(predictor, suffix)
    base_path = str(resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv'))
    base_path = base_path + 'ensemble_models'
    if not os.path.exists(base_path):
        print('Ensemble models do not exist now, training...')
        ensemble_maker(predictor)
    print(('ANN activated for ' + str(predictor)))
    model_list = glob.glob(base_path + '/*.h5')

    labels = load_test_labels(predictor)
    if 'clf' not in predictor:
        labels = np.array(labels, dtype='float64')
    else:
        labels = np.array(labels, dtype='int')
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)
    if (descriptors and descriptor_names):
        excitation = tf_ANN_excitation_prepare(predictor, descriptors, descriptor_names)
        excitation = data_normalize(excitation, train_mean_x, train_var_x)
    else:
        mat = load_test_data(predictor)
        mat = np.array(mat, dtype='float64')
        train_mat = data_normalize(mat, train_mean_x, train_var_x)
        excitation = np.array(train_mat)
    print(('excitation is ' + str(excitation.shape)))
    print(('actual label:', labels[:3]))
    results_list = []
    # print('models', model_list)
    for idx, model in enumerate(model_list):
        _base = model.split('.')[0]
        with open(_base + '.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into  model
        loaded_model.load_weights(model)
        # complile model
        loaded_model.compile(loss="mse", optimizer='adam',
                             metrics=['mse', 'mae', 'mape'])
        if 'clf' not in predictor:
            result = data_rescale(loaded_model.predict(excitation), train_mean_y, train_var_y)
        else:
            result = loaded_model.predict(excitation)
            result = np.squeeze(result, axis=1)
        results_list.append(result)
    results_list = np.transpose(np.array(results_list))
    # print(results_list.shape)
    result_mean, result_std = np.mean(results_list, axis=1), np.std(results_list, axis=1)
    labels = np.squeeze(labels, axis=1)
    print((labels.shape, result_mean.shape))
    error_for_mean = np.abs(labels - result_mean)
    return result_mean, result_std, error_for_mean


def mc_dropout_uq(predictor, descriptors=False, descriptor_names=False, num=500):
    labels = load_test_labels(predictor)
    if 'clf' not in predictor:
        labels = np.array(labels, dtype='float64')
    else:
        labels = np.array(labels, dtype='int')
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)
    if (descriptors and descriptor_names):
        excitation = tf_ANN_excitation_prepare(predictor, descriptors, descriptor_names)
        excitation = data_normalize(excitation, train_mean_x, train_var_x)
    else:
        mat = load_test_data(predictor)
        mat = np.array(mat, dtype='float64')
        train_mat = data_normalize(mat, train_mean_x, train_var_x)
        excitation = np.array(train_mat)
    print(('excitation is ' + str(excitation.shape)))
    loaded_model = load_keras_ann(predictor)
    get_outputs = K.function([loaded_model.layers[0].input, K.learning_phase()],
                             [loaded_model.layers[-1].output])
    print(('LOADED MODEL HAS ' + str(
        len(loaded_model.layers)) + ' layers, so latent space measure will be from first ' + str(
        len(loaded_model.layers) - 1) + ' layers'))
    results_list = []
    err_list = []
    for ii in range(num):
        if not np.mod(ii, int(num / 10)):
            print(('%d / %d' % (ii, num)))
        if 'clf' not in predictor:
            results = data_rescale(np.array(get_outputs([excitation, 1])), train_mean_y,
                                   train_var_y)[0]
        else:
            results = np.array(get_outputs([excitation, 1]))[0]
        results = results.squeeze(axis=1)
        err = np.linalg.norm(labels - results) ** 2
        results_list.append(results)
        err_list.append(err)
    results_list = np.transpose(np.array(results_list))

    def f(tau):
        return mc_dropout_logp(tau, np.array(err_list))
    tau = sp.optimize.minimize(f, 10).x
    result_mean, result_std = np.mean(results_list, axis=1), np.std(results_list, axis=1)
    result_std = np.sqrt(1 / tau + result_std ** 2)
    # print(tau, result_std[:3])
    labels = np.squeeze(labels, axis=1)
    error_for_mean = np.abs(labels - result_mean)
    return result_mean, result_std, error_for_mean


def latent_space_uq(predictor, layer_index=-2, descriptors=False, descriptor_names=False, entropy=False):
    key = get_key(predictor, suffix=False)
    base_path = str(resource_files("molSimplify.tf_nn").joinpath(f'{key}.csv'))
    base_path = base_path + 'ensemble_models'
    if not os.path.exists(base_path):
        print('Ensemble models do not exist now, training...')
        ensemble_maker(predictor)
    print(('ANN activated for ' + str(predictor)))
    model_list = glob.glob(base_path + '/*.h5')
    train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(predictor)
    if (descriptors and descriptor_names):
        excitation = tf_ANN_excitation_prepare(predictor, descriptors, descriptor_names)
        excitation = data_normalize(excitation, train_mean_x, train_var_x)
    else:
        mat = load_training_data(predictor)
        mat = np.array(mat, dtype='float64')
        train_mat = data_normalize(mat, train_mean_x, train_var_x)
        excitation = np.array(train_mat)
    ### load test data
    if (descriptors and descriptor_names):
        excitation = tf_ANN_excitation_prepare(predictor, descriptors, descriptor_names)
        excitation_test = data_normalize(excitation, train_mean_x, train_var_x)
    else:
        mat = load_test_data(predictor)
        mat = np.array(mat, dtype='float64')
        test_mat = data_normalize(mat, train_mean_x, train_var_x)
        excitation_test = np.array(test_mat)
    labels = load_test_labels(predictor)
    labels_train = load_training_labels(predictor)
    if 'clf' not in predictor:
        labels = np.array(labels, dtype='float64')
        labels_train = np.array(labels_train, dtype='float64')
    else:
        labels = np.array(labels, dtype='int')
        labels_train = np.array(labels_train, dtype='int')
    results_list = []
    err_list = []
    dist_list = []
    for model in model_list:
        _base = model.split('.')[0]
        with open(_base + '.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model)
        loaded_model.compile(loss="mse", optimizer='adam',
                             metrics=['mse', 'mae', 'mape'])
        get_outputs = K.function([loaded_model.layers[0].input, K.learning_phase()],
                                 [loaded_model.layers[-1].output])
        get_latent = K.function([loaded_model.layers[0].input, K.learning_phase()],
                                [loaded_model.layers[layer_index].output])
        print(('NOTE: you are choosing:', loaded_model.layers[layer_index], loaded_model.layers[layer_index].name,
              'for the latence space!'))
        if 'clf' not in predictor:
            results = data_rescale(np.array(get_outputs([excitation_test, 0])), train_mean_y,
                                   train_var_y)[0]
        else:
            results = np.array(get_outputs([excitation_test, 0]))[0]
        ## get latent dist
        training_latent_distance = np.array(get_latent([excitation, 0]))[0]
        nn_latent_dist_train, _, __ = dist_neighbor(training_latent_distance, training_latent_distance, labels_train,
                                                    l=5, dist_ref=1)
        nn_dist_avrg_train = np.mean(nn_latent_dist_train)
        # print(nn_dist_avrg_train)
        test_latent_distance = np.array(get_latent([excitation_test, 0]))[0]
        nn_latent_dist_test, nn_dists, nn_labels = dist_neighbor(test_latent_distance, training_latent_distance,
                                                                 labels_train,
                                                                 l=5, dist_ref=nn_dist_avrg_train)
        if not entropy:
            # print(nn_latent_dist_test.shape)
            # print(min(nn_latent_dist_test), max(nn_latent_dist_test))
            dist_list.append(nn_latent_dist_test)
        else:
            entropy = []
            for idx, _dists in enumerate(nn_dists):
                entropy.append(get_entropy(_dists, nn_labels[idx]))
            dist_list.append(np.array(entropy))
        results = results.squeeze(axis=1)
        err = np.linalg.norm(labels - results) ** 2
        results_list.append(results)
        err_list.append(err)
    dist_list = np.transpose(np.array(dist_list))
    results_list = np.transpose(np.array(results_list))
    result_mean = np.mean(results_list, axis=1)
    latent_dist = np.mean(dist_list, axis=1)
    labels = np.squeeze(labels, axis=1)
    error_for_mean = np.abs(labels - result_mean)
    return result_mean, latent_dist, error_for_mean

###########
# predictor = 'geo_static_clf'
# ensemble_maker(predictor, num=10)
# _result_mean, _result_std, _error_for_mean = ensemble_uq(predictor)
# result_mean, result_std, error_for_mean = mc_dropout_uq(predictor)
# result_mean, latent_dist, error_for_mean = latent_space_uq(predictor, layer_index=-4, entropy=True)

# stds = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35]  # for variance
# stds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8]  # for latent distances
# stds = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]  # for entropy
# plot_scatter(latent_dist, error_for_mean, xlabel='entropy', ylabel='prediction error',
#              figname='entropy_err.pdf')
# plot_dist_err(pred_std=latent_dist, pred_err=error_for_mean, stds=stds,
#               label_x='entropy', label_y='accuracy',
#               lable_y2='ratio of data', figname='entropy_acc_ratio.pdf')
# plot_metrics_correlation(metric1=latent_dist, metric2=_result_std, pred_err=error_for_mean,
#                          xlabel='entropy', ylabel='ensemble std', figname='dist_relations_entropy_ensemblestd.pdf')
