from ml_p.blueprints import Pipe
from ml_p.blueprints import NoFitError
import numpy as np
import numpy.linalg as ln
from scipy.linalg import eigh


class Pca(Pipe):

    def __init__(self, n_dim=None):
        self.X = None
        self.mu = None
        self.DC = None
        self.COV = None
        self.P = None
        self.n_dim = n_dim

    def fit(self, x: np.array, y=None) -> None:
        if self.n_dim is None:
            self.n_dim = x.shape[1]
        self.X = x
        self.mu = self.X.mean(axis=0)
        self.DC = self.X - self.mu
        self.COV = (self.DC.T @ self.DC) / self.X.shape[0]
        s, U = ln.eigh(self.COV)
        self.P = U[:, ::-1][:, 0:self.n_dim]
        pass

    def fit_transform(self, x, y=None):
        self.fit(x, None)
        return self.transform(x)

    def transform(self, x):
        if x is None:
            raise NoFitError()
        return (self.P.T @ x.T).T


class Lda(Pipe):

    def __init__(self, n_dim=None):
        self.X = None
        self.Y = None
        self.m = n_dim
        self.U = None

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x):
        return (self.U.T @ x.T).T

    def fit(self, x, y):
        if self.m is None:
            self.m = x.shape[0]
        self.X = x.T
        self.Y = y

        mu = self.X.mean(1).reshape((-1, 1))
        SB_ls = []
        SW_ls = []
        for label in set(y):
            x_c = self.X[:, y == label]
            mu_c = x_c.mean(1).reshape((-1, 1))
            SB_ls.append(x_c.shape[1]*((mu_c-mu) @ (mu_c-mu).T))
            SW_ls.append((x_c-mu_c) @ (x_c-mu_c).T)
        SB = sum(SB_ls)/self.X.shape[1]
        SW = sum(SW_ls)/self.X.shape[1]
        s, U = eigh(SB, SW)
        # W = U[:, ::-1][:, :self.m]
        # UW, _, _ = ln.svd(W)
        # self.U = UW[:, :self.m]
        self.U = U[:, ::-1][:, :self.m]




