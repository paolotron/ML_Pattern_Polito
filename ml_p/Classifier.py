import scipy.special
from .blueprints import *
import numpy as np
from .probability import GAU_ND_logpdf
from .preproc import get_cov
import ml_p


class Perceptron(Faucet):

    def __init__(self, iterations=100, alpha=0.1, seed=None):
        self.Y = None
        self.X = None
        self.labels = None
        self.iter = iterations
        self.alpha = alpha
        self.weights = None
        self.seed = seed

    def fit(self, x, y):
        if self.seed is not None:
            np.random.seed(self.seed)

        rand = np.random.choice(x.shape[0], x.shape[0])
        self.X = np.hstack([np.ones((x.shape[0], 1)), x])[rand, :].T
        self.Y = y.reshape((-1, 1))[rand, :]
        self.labels = np.unique(y)

        def single(y_true):
            w = np.random.random((self.X.shape[0], 1)).T - 0.5
            for i in range(self.iter):
                for data, y_tr in zip(self.X.T, y_true.T):
                    data = data.reshape((1, -1))
                    y_predict = np.heaviside(w @ data.T, 0)
                    w = w + self.alpha * (y_tr - y_predict) * data
            return w

        lis = []
        for label in self.labels:
            y_l = (self.Y == label)
            lis.append(single(y_l.T))
        self.weights = np.vstack(lis).T
        return self

    def predict(self, x):
        one = np.ones((x.shape[0], 1))
        x = np.hstack([one, x]).T
        y = self.weights.T @ x
        return self.labels[np.argmax(y, axis=0)]

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)


class GaussianClassifier(Faucet):

    def __init__(self):
        self.estimates = []
        self.labels = None
        self.Spost = None

    def fit(self, x, y):
        un, con = np.unique(y, return_counts=True)
        for label, count in zip(un, con):
            matrix = x[y == label, :]
            self.estimates.append((label, np.mean(matrix, 0), ml_p.preproc.get_cov(matrix), count / x.shape[0]))
        return self

    def predict(self, x):
        scores = []
        for label, mu, cov, prob in self.estimates:
            scores.append(ml_p.probability.GAU_ND_logpdf(x.T, mu.reshape(-1, 1), cov) + np.log(prob))
        SJoint = np.hstack([value.reshape(-1, 1) for value in scores])
        logsum = scipy.special.logsumexp(SJoint, axis=1)
        self.Spost = SJoint - logsum.reshape(1, -1).T
        res = np.argmax(self.Spost, axis=1)
        return res

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)


class NaiveBayes(GaussianClassifier):

    def fit(self, x, y):
        un, con = np.unique(y, return_counts=True)
        for label, count in zip(un, con):
            matrix = x[y == label, :]
            cov = np.diag(np.var(matrix, 0))
            self.estimates.append((label, np.mean(matrix, 0), cov, count / x.shape[0]))
        return self


class TiedGaussian(GaussianClassifier):

    def fit(self, x, y):
        super().fit(x, y)
        sigma = (1/y.shape[0])*sum([sigma*sum(y == label)for label, mu, sigma, prob in self.estimates])
        self.estimates = [(label, mu, sigma, prob) for label, mu, _, prob in self.estimates]
        return self
