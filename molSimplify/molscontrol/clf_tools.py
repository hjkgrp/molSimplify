import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from packaging import version
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.neighbors import BallTree
from tensorflow.keras import Model

"""
tools for ML models.
"""


def get_layer_outputs(model, layer_index, input,
                      training_flag=False):
    if not version.parse(tf.__version__) >= version.parse('2.0.0'):
        get_outputs = K.function([model.layers[0].input, K.learning_phase()],
                                 [model.layers[layer_index].output])
        nn_outputs = get_outputs([input, training_flag])[0]
    else:
        partial_model = Model(model.inputs, model.layers[layer_index].output)
        nn_outputs = partial_model([input], training=training_flag).numpy()  # runs the model in training mode
    return nn_outputs


def _dist_neighbor(fmat1, fmat2, labels, l=5, dist_ref=1):  # noqa E741
    dist_mat = pairwise_distances(fmat1, fmat2, 'manhattan')
    dist_mat = dist_mat * 1.0 / dist_ref
    dist_avrg, dist_list, labels_list = [], [], []
    for ele in dist_mat:
        dist_arr = np.round(np.array(ele), 4)
        if not dist_ref == 1:
            _count = (dist_arr < 10).sum()
            _count = l if _count < l else _count
            _count = _count if _count < 300 else 300
        else:
            _count = l
        ind = dist_arr.argsort()[:_count]
        _dist = dist_arr[ind]
        dist_list.append(_dist)
        _labels = np.array([labels[x] for x in ind])
        labels_list.append(_labels)
        if _dist.all() > 1e-4:
            dist_avrg.append(np.mean(_dist[:l]))
        else:
            dist_avrg.append(np.mean(_dist[:l]) * float(l) / (l - 1))
    # print('-----mean: %f, std: %f---' % (np.mean(dist_avrg), np.std(dist_avrg)))
    dist_avrg = np.array(dist_avrg)
    dist_list = np.array(dist_list)
    labels_list = np.array(labels_list)
    return dist_avrg, dist_list, labels_list


def dist_neighbor(fmat1, fmat2, labels, l=10, dist_ref=1):  # noqa E741
    tree = BallTree(fmat2, leaf_size=2, metric='cityblock')
    dist_mat, inds = tree.query(fmat1, l)
    dist_mat = dist_mat * 1.0 / dist_ref
    dist_avrg = np.mean(dist_mat, axis=1)
    labels_list = labels[inds]
    return dist_avrg, dist_mat, labels_list


def _get_entropy(dists, neighbor_targets):
    entropies = []
    _sum = 0
    for ii, _neighbor_targets in enumerate(neighbor_targets):
        p0, p1 = dist_penalty(2), dist_penalty(2)
        for idx, tar in enumerate(_neighbor_targets):
            tar = int(tar)
            d = dists[ii][idx]
            if d <= 10:
                if d != 0:
                    if tar == 0:
                        p0 += dist_penalty(d)
                    elif tar == 1:
                        p1 += dist_penalty(d)
                else:
                    if tar == 0:
                        p0 += 100
                    elif tar == 1:
                        p1 += 100
            _sum = p0 + p1
        p0 = p0 / _sum
        p1 = p1 / _sum
        if p1 == 0 or p0 == 0:
            entropies.append(0)
        else:
            entropies.append(-(p0 * np.log(p0) + p1 * np.log(p1)))
    return np.array(entropies)


def get_entropy(dists, neighbor_targets, nclasses=2):
    entropies = []
    for ii, _neighbor_targets in enumerate(neighbor_targets):
        p = [dist_penalty(2) for ii in range(nclasses)]
        for idx, tar in enumerate(_neighbor_targets):
            tar = int(tar)
            d = dists[ii][idx]
            if d <= 10:
                p[tar] += dist_penalty(d) if d > 1e-6 else 100
        p = [x/np.sum(p) for x in p]
        _entropy = 0
        for ii in range(nclasses):
            _entropy += -p[ii] * np.log(p[ii])
        entropies.append(_entropy)
    return np.array(entropies)


def dist_penalty(d):
    return np.exp(-1 * d ** 2)


def find_closest_model(step, allowed_steps):
    step_chosen = 0
    for _s in allowed_steps:
        delta = step - _s
        if ("mindelta" not in list(locals().keys())):
            mindelta = abs(delta)
            step_chosen = _s
        elif (abs(delta) < mindelta):
            mindelta = abs(delta)
            step_chosen = _s
    return step_chosen, mindelta


def get_lsd(model, X_train, X, labels_train=None):
    '''
    Get latent space distance
    inputs:
        model: ANN model
        X_train: np.array, training features
        X: features for target point(s)
        labels_train: np.array, names for training poinst (optional)
    outputs:
        dist_avrg: np.array, lsd for target points
    '''
    if labels_train is None:
        labels_train = np.array([0 for _ in X_train])
    ls_train = get_layer_outputs(model, layer_index=-3, input=X_train, training_flag=False)
    ls = get_layer_outputs(model, layer_index=-3, input=X, training_flag=False)
    _dist_avrg, _, _ = dist_neighbor(fmat1=ls_train, fmat2=ls_train, labels=labels_train, l=10, dist_ref=1)
    avrg_ls_train = np.mean(_dist_avrg)
    dist_avrg, _, _ = dist_neighbor(fmat1=ls, fmat2=ls_train, labels=labels_train, l=10, dist_ref=avrg_ls_train)
    return dist_avrg
