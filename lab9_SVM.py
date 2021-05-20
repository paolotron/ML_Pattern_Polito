import ml_p.validation as val
from lab8_logistic import load_iris_binary
from ml_p.Classifier import SVM


def main():
    data, label = load_iris_binary()
    DTR, LTR, DTE, LTE = val.train_test_split(data, label, 2 / 3)
    for hyper in [(1, 0.1), (1, 1.), (1, 10.), (10, 0.1), (10, 1.), (10, 10.)]:
        model = SVM(*hyper)
        model.fit(DTR, LTR)
        prediction = model.predict(DTE)
        print(f"K: {hyper[0]} C: {hyper[1]} PrimalLoss: {model.primal} DualityGap: {model.get_gap()} error rate: {val.err_rate(prediction, LTE)}")

    for hyper in [(0, 1, "Poly", (2, 0)), (1, 1, "Poly", (2, 0)), (0, 1, "Poly", (2, 1)), (1, 1, "Poly", (2, 1)),
                  (0, 1, "Radial", (1,)), (0, 1, "Radial", (10,)), (1, 1, "Radial", (1,)), (1, 1, "Radial", (10,))]:
        model = SVM(*hyper)
        model.fit(DTR, LTR)
        prediction = model.predict(DTE)
        print(f"K: {hyper[0]}, C: {hyper[1]}, Ker: {hyper[2]}, {hyper[3]}, err: {val.err_rate(prediction, LTE)}")


if __name__ == "__main__":
    main()
