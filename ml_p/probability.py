import numpy as np
import numpy.linalg as ln


def GAU_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """
    Probability function of Guassian distribution
    :param x: ndarray input parameters
    :param mu: float mean of the distribution
    :param var: float variance of the distribution
    :return: ndarray probability of each sample
    """
    k = (1 / (np.sqrt(2 * np.pi * var)))
    up = -np.power(x - mu, 2) / (2 * var)
    return k * np.exp(up)


def GAU_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """
    Log probability function of Guassian distribution
    :param x: ndarray input parameters
    :param mu: float mean of the distribution
    :param var: float variance of the distribution
    :return: ndarray log probability of each sample
    """
    return -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var) - np.power(x - mu, 2) / (2 * var)


def GAU_ND_logpdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Multivarate Gaussian Distribution probability function<
    :param x: ndarray input matrix
    :param mu: ndarray mean vector
    :param cov: ndarray covariance matrix
    :return: ndarray
    """
    M = x.shape[0]
    s, ld = ln.slogdet(cov)
    k = -M * np.log(2 * np.pi) * 0.5 - s * ld * 0.5
    f = x - mu
    res = k - .5 * ((f.T @ ln.inv(cov)).T * f).T.sum(-1)
    return res
