from blueprints import *
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
        predictions = []
        for data in x:
            for class_ in self.classes.items():
                prob_class = class_[1]
                pass

    def fit_predict(self, x, y):
        pass
