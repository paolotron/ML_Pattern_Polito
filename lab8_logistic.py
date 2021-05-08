import numpy as np
import scipy.optimize as opt
import sklearn.datasets
from ml_p import validation
from ml_p.Classifier import LogisticRegression


def test_numerical_minimum():
    def fun(arr):
        return np.power(arr[0] + 3, 2) + np.sin(arr[0]) + np.power(arr[1] + 1, 2)

    def grad_fun(arr):
        fx = fun(arr)
        grad = np.array([2 * (arr[0] + 3) + np.cos(arr[0]), 2 * (arr[1] + 1)])
        return fx, grad

    x1, f1, d1 = opt.fmin_l_bfgs_b(fun, np.array([0, 0]), approx_grad=True)
    print("Minimum with approximated gradient:", x1, "\nMinimum value: ", f1)
    x2, f2, d2 = opt.fmin_l_bfgs_b(grad_fun, np.array([0, 0]), approx_grad=False)
    print("Minimum with precise gradient:", x2, "\nMinimum value: ", f2)
    print(d1, "\n", d2)


# noinspection DuplicatedCode
def binary_logistic():
    data, label = load_iris_binary()
    DTR, LTR, DTE, LTE = validation.train_test_split(data, label, 2 / 3)

    for regular in (0, float("10e-6"), float("10e-3"), 1):
        model = LogisticRegression(norm_coeff=regular)
        model.fit(DTR, LTR)
        print("Lambda:", regular, " err_rate: ", validation.err_rate(model.predict(DTE), LTE))


# noinspection DuplicatedCode
def multiclass_logistic():
    data, label = sklearn.datasets.load_iris()['data'], sklearn.datasets.load_iris()['target']
    DTR, LTR, DTE, LTE = validation.train_test_split(data, label, 2 / 3)

    for regular in (0, float("1e-6"), float("1e-3"), 1):
        model = LogisticRegression(norm_coeff=regular)
        model.fit(DTR, LTR)
        print("Lambda:", regular, " err_rate: ", validation.err_rate(model.predict(DTE), LTE))


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'], sklearn.datasets.load_iris()['target']
    D = D[L != 0, :]
    L = L[L != 0]
    L[L == 2] = 0
    return D, L


if __name__ == "__main__":
    multiclass_logistic()