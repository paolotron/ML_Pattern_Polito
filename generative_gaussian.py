import sklearn.datasets
from ml_p.validation import train_test_split
from ml_p.validation import accuracy
from ml_p.validation import err_rate
import numpy as np
import ml_p.preproc

import ml_p.Classifier as Cls
from ml_p.validation import leave_one_out_split


def load():
    return sklearn.datasets.load_iris()['data'], sklearn.datasets.load_iris()['target']


def test_kfold(D, L):
    models = [Cls.GaussianClassifier, Cls.NaiveBayes, Cls.TiedGaussian]
    error_rates = [0, 0, 0, 0]
    n = D.shape[0]
    for x_tr, y_tr, x_ts, y_ts in leave_one_out_split(D, L):
        error_rates = [err_rate(mod().fit(x_tr, y_tr).predict(x_ts), y_ts)/n + error for mod, error in zip(models, error_rates)]
    print(error_rates)


def lab5():
    D, L = load()
    DTR, LTR, DTE, LTE = train_test_split(D, L, 2 / 3)
    test_kfold(D, L)
    model = ml_p.Classifier.TiedGaussian()
    model.fit(DTR, LTR)
    res = model.predict(DTE)
    return res, LTE, model


if __name__ == "__main__":
    res, LTE, model = lab5()
    print(accuracy(res, LTE))
    print(np.sum(np.exp(model.Spost),axis=1))
