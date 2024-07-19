from typing import Callable
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from packaging import version
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model


def get_acc(pred_std, pred_err, stds):
    pred_err_arr = []
    for target_std in stds:
        _pred_err = []
        for idx, _std in enumerate(pred_std):
            if _std < target_std:
                _pred_err.append(pred_err[idx])
        pred_err_arr.append(_pred_err)
    acc, ratio = [], []
    for idx, _ in enumerate(pred_err_arr):
        pred_err_now = pred_err_arr[idx]
        acc_arr = [1 if pred_err_now[ii] < 0.5 else 0 for ii in range(len(pred_err_now))]
        acc.append(np.mean(acc_arr))
        ratio.append(1.0 * len(pred_err_now) / len(pred_std))
    print(('stds', stds))
    print(('accuracy', acc))
    print(('ratio', ratio))
    return stds, np.array(acc), np.array(ratio)


def dist_neighbor(fmat1, fmat2, labels, l=5, dist_ref=1):  # noqa: E741
    dist_mat = pairwise_distances(fmat1, fmat2, 'manhattan')
    dist_mat = dist_mat * 1.0 / dist_ref
    dist_avrg, dist_list, labels_list = [], [], []
    # print('shape of dist_mat:', dist_mat.shape)
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
        _labels = np.array([labels[x][0] for x in ind])
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


def array_stack(array, _idx):
    for ii in range(len(array)):
        if not ii == _idx:
            if 'out_arr' not in dir():
                out_arr = array[ii]
            else:
                out_arr = np.concatenate((out_arr, array[ii]))
    return out_arr


def dist_penalty(d):
    return np.exp(-1 * d ** 2)


def get_entropy(dists, neighbor_targets) -> np.ndarray:
    entropies = []
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


def get_layer_outputs(model, layer_index, input,
                      training_flag=False) -> np.ndarray:
    if not version.parse(tf.__version__) >= version.parse('2.0.0'):
        get_outputs = K.function([model.layers[0].input, K.learning_phase()],
                                 [model.layers[layer_index].output])
        nn_outputs = get_outputs([input, training_flag])[0]
    else:
        partial_model = Model(model.inputs, model.layers[layer_index].output)  # type: Callable
        nn_outputs = partial_model([input], training=training_flag).numpy()  # runs the model in training mode
    return nn_outputs


def lse_trust(lse):
    level = False
    if lse < 0.2:
        level = 'very high'
    elif lse < 0.3:
        level = 'high'
    elif lse < 0.4:
        level = 'medium'
    elif lse < 0.5:
        level = 'low'
    else:
        level = 'very low'
    return level
