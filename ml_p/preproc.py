from ml_p.blueprints import Pipe
from ml_p.blueprints import NoFitError
import numpy.linalg as ln
from scipy.linalg import eigh


class Pca(Pipe):

    def __init__(self, n_dim=None, center=False):
        """
        Pca init method
        :param n_dim: int number of dimensions of the output data
        :param center: Bool default False, if True the output is centered
        """
        self.X = None
        self.P = None
        self.n_dim = n_dim
        self.mu = None
        self.center = center

    def fit(self, x, y=None):
        """
        calculate the P matrix through eig-decomposition
        of the Covariance Matrix
        :param x: array-like
        :param y: None, just for compatibility
        :return: None
        """
        if self.n_dim is None:
            self.n_dim = x.shape[1]
        self.X = x
        mu = self.X.mean(axis=0)
        self.mu = mu
        DC = self.X - mu
        COV = (DC.T @ DC) / self.X.shape[0]
        s, U = ln.eigh(COV)
        self.P = U[:, ::-1][:, 0:self.n_dim]

    def fit_transform(self, x, y=None):
        """
        Compute fit and transform on the same input data
        :param x: array-like
        :param y: None
        :return: array-like
        """
        self.fit(x, None)
        return self.transform(x)

    def transform(self, x):
        """
        Transform the data with the P computed already
        :param x: array-like
        :return: array-like
        """
        if x is None:
            raise NoFitError()
        return (self.P.T @ (x - (self.mu if self.center else 0)).T).T


class Lda(Pipe):

    def __init__(self, n_dim=None, center=False):
        """
        Lda init method
        :param n_dim: int number of dimensions of the output data
        :param center: Bool default False, if True the output is centered
        """
        self.X = None
        self.Y = None
        self.m = n_dim
        self.U = None
        self.mu = None
        self.center = center

    def transform(self, x):
        """
        Compute the output through the U matrix on the input data
        :param x:
        :return: array-like
        """
        return (self.U.T @ (x - (self.mu.T if self.center else 0)).T).T

    def fit(self, x, y):
        """
        Compute the U matrix with the between and in-between classes covariance matrixes
        :param x: array-like
        :param y: array-like, classes of x
        :return: None
        """
        if self.m is None:
            self.m = x.shape[0]
        self.X = x.T
        self.Y = y

        mu = self.X.mean(1).reshape((-1, 1))
        self.mu = mu
        SB_ls = []
        SW_ls = []
        for label in set(y):
            x_c = self.X[:, y == label]
            mu_c = x_c.mean(1).reshape((-1, 1))
            nc = x_c.shape[1]
            SB_ls.append(nc * (mu_c-mu) @ (mu_c-mu).T)
            SW_ls.append((x_c-mu_c) @ (x_c-mu_c).T)

        N = self.X.shape[1]
        SB = sum(SB_ls) / N  # Between Class Variability Matrix
        SW = sum(SW_ls) / N  # Within Class Variability Matrix

        s, U = eigh(SB, SW)
        self.U = U[:, ::-1][:, :self.m]


class StandardScaler(Pipe):

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mu = None
        self.std = None

    def fit(self, x, y=None):
        self.mu = x.mean(0)
        self.std = x.std(0)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x):
        mu = self.mu if self.with_mean else 0
        std = self.std if self.with_std else 1
        return (x - mu) / std




