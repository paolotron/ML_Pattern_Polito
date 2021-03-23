from .blueprints import *
import numpy as np


class NaiveBayes(Faucet):

    def __init__(self):
        self.X = None
        self.Y = None
        self.classes = None

    def fit(self, x, y):
        self.X = x
        self.Y = y
        self.classes = dict(np.unique(y, return_counts=True))

    def predict(self, x):
        pass

    def fit_predict(self, x, y):
        pass


class Perceptron(Faucet):

    def __init__(self, iterations=100, alpha=0.1):
        self.Y = None
        self.X = None
        self.labels = None
        self.iter = iterations
        self.alpha = alpha
        self.weights = None

    def fit(self, x, y):
        rand = np.random.choice(x.shape[0], x.shape[0])
        self.X = np.hstack([ np.ones((x.shape[0], 1)), x])[rand, :].T
        self.Y = y.reshape((-1, 1))[rand, :]
        self.labels = np.unique(y)

        def single(y_true):
            w = np.random.random((self.X.shape[0], 1)).T-0.5
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

    def predict(self, x):
        one = np.ones((x.shape[0], 1))
        x = np.hstack([one, x]).T
        y = self.weights.T @ x
        return self.labels[np.argmax(y, axis=0)]

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)
