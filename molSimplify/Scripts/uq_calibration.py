from functools import partial
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def nlm(func, p0,
        dists, epsilons,
        varfun):
    '''
    non-linear optimization.
    Usage: nlm(maximum_likelihood_loss, [1, 2], dists, epsilons, varfun=linear_variance_model)

    :param func: objective function to be minimized
    :param p0: initial guess of the parameters
    :param dists: latent space distance (or other uq metric)
    :param epsilons: errors for predictions
    :param varfun: variance function for the final Gaussian distribution
    :return: best parameters
    '''
    objective_func = partial(func,
                             dists=dists,
                             epsilons=epsilons,
                             varfun=varfun)
    m = minimize(objective_func, p0,
                 method='Powell',
                 jac=False,
                 tol=1e-4,
                 options={'maxiter': 1000, 'disp': True})
    return m.x


def linear_variance_model(dist, theta):
    '''

    :param dist: latent space distance (or other uq metric)
    :param theta: parameters for the condition Gaussian distribution err(d) ~ N(0, theta_0^2 + d* theta_1^2)
    :return: std for the conditional Gaussian distribution
    '''
    return theta[0] ** 2 + (theta[1] ** 2) * dist


def zero_linear_variance_model(dist, theta):
    '''

    :param dist: latent space distance (or other uq metric)
    :param theta: parameters for the condition Gaussian distribution err(d) ~ N(0, d* theta_0^2)
    :return: std for the conditional Gaussian distribution
    '''
    return (theta[0] ** 2) * dist


def maximum_likelihood_loss(theta, dists, epsilons, varfun=linear_variance_model):
    '''

    :param theta: parameters for the condition Gaussian distribution err(d) ~ N(0, theta_0^2 + d* theta_1^2)
    :param dists: latent space distance (or other uq metric)
    :param epsilons: errors for predictions
    :param varfun: variance function for the final Gaussian distribution
    :return: loss of the maximum likelihood.
    '''
    L = 0
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        dist = dists[i]
        this_var = varfun(dist, theta)
        L += -np.log(norm.pdf(epsilon, 0, np.sqrt(this_var)))
        # L += 0.5 * np.log(2 * np.pi * np.sqrt(this_var)) + 1. * (epsilon ** 2) / (2 * this_var)
    return L
