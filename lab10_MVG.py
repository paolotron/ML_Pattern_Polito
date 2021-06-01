import numpy as np
import json
from ml_p.probability import logpdf_GMM, LGB_estimation
from ml_p.probability import EM_estimation
from ml_p.Classifier import GaussianMixture
import ml_p.validation
import sklearn.datasets


def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)


def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]


def test_logpdf_GMM():
    data = np.load("datasets/lab10/GMM_data_4D.npy")
    reference = load_gmm("datasets/lab10/GMM_4D_3G_init.json")
    act_result = np.load("datasets/lab10/GMM_4D_3G_init_ll.npy")
    res = logpdf_GMM(data, reference)
    assert (res == act_result).all()


def test_GMM_estimation():
    data = np.load("datasets/lab10/GMM_data_4D.npy")
    reference = load_gmm("datasets/lab10/GMM_4D_3G_init.json")
    sol = load_gmm("datasets/lab10/GMM_4D_3G_EM.json")
    new_gmm = EM_estimation(data, reference)
    print("TEST GMM:")
    print(np.mean(logpdf_GMM(data, new_gmm)))
    print(np.mean(logpdf_GMM(data, sol)))


def test_LGB_alg():
    data1d = np.load("datasets/lab10/GMM_data_1D.npy")
    reference1d = load_gmm("datasets/lab10/GMM_1D_4G_EM_LBG.json")
    res = LGB_estimation(data1d, 0.1, 2)
    print("TEST LGB:")
    print(np.mean(logpdf_GMM(data1d, reference1d)))
    print(np.mean(logpdf_GMM(data1d, res)))


def test_LGB_psi():
    print("TEST LGB PSI:")
    data = np.load("datasets/lab10/GMM_data_4D.npy")
    res = LGB_estimation(data, 0.1, 2, psi=1)
    print(np.mean(logpdf_GMM(data, res)))


def test_tied_GMM():
    print("TEST TIED GMM")
    data = np.load("datasets/lab10/GMM_data_4D.npy")
    res = LGB_estimation(data, 0.1, 2, tied=True)
    print(np.mean(logpdf_GMM(data, res)))


def test_classification():
    print("TEST LABELING")
    D, L = sklearn.datasets.load_iris()['data'], sklearn.datasets.load_iris()['target']
    DTR, LTR, DTE, LTE = ml_p.validation.train_test_split(D, L, 2 / 3)
    hypers1 = [{"alpha": 0.01, "N": i, "psi": 0.01} for i in range(5)]
    hypers2 = [{"alpha": 0.01, "N": i, "psi": 0.01, "diag": True} for i in range(5)]
    hypers3 = [{"alpha": 0.01, "N": i, "psi": 0.01, "tied": True} for i in range(5)]
    for hyper_set, name in zip([hypers1, hypers2, hypers3], ["Full Covariance", "Diagonal Covariance", "Tied Covariance"]):
        print(name, end="")
        for hyper in hyper_set:
            model = GaussianMixture(**hyper)
            model.fit(DTR, LTR)
            print(" {:.2f} ".format(ml_p.validation.err_rate(model.predict(DTE), LTE)), end="")
        print("")


def main():
    # test_logpdf_GMM()
    # test_GMM_estimation()
    # test_LGB_alg()
    # test_LGB_psi()
    # test_tied_GMM()
    test_classification()


if __name__ == "__main__":
    main()
