import numpy as np

import generative_gaussian
import ml_p.validation as valid
from ml_p.probability import optimal_bayes_decision_with_ratio
from ml_p.probability import bayes_detection_function_with_confusion
from ml_p.probability import optimal_bayes_decision_with_threshold
from ml_p.probability import minimal_detection_cost
from scipy.special import logsumexp
import matplotlib.pyplot as plt


def confusion_iris():
    res_label, real_label, _ = generative_gaussian.lab5()
    conf = valid.confusion_matrix(res_label, real_label)
    print(np.array2string(conf))


def confusion_dante():
    res_label, real_label = np.argmax(np.load("datasets/lab7/commedia_ll.npy"), axis=0), \
                            np.load("datasets/lab7/commedia_labels.npy")
    conf = valid.confusion_matrix(res_label, real_label)
    print(np.array2string(conf))


def test_optimal_decision():
    for parameters in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:
        print("parameters: ", parameters)

        posterior_prob, false_negative_cost, false_positive_cost = parameters
        datall = np.load("datasets/lab7/commedia_llr_infpar.npy")
        labels_real = np.load("datasets/lab7/commedia_labels_infpar.npy")
        labels_calc = optimal_bayes_decision_with_ratio(datall, posterior_prob, false_negative_cost,
                                                        false_positive_cost)
        confusion = valid.confusion_matrix(labels_calc, labels_real)
        print("Confusion matrix:\n", confusion)
        risk = bayes_detection_function_with_confusion(confusion, posterior_prob, false_negative_cost,
                                                       false_positive_cost)
        print("Risk: ", risk)
        normalizer_factor = min(posterior_prob * false_negative_cost, (1 - posterior_prob) * false_positive_cost)
        print("Normalized Risk:",
              risk / normalizer_factor)

        # TODO discover how to do minDCF

        res = []
        for t in np.linspace(min(datall), max(datall), 1000):
            lab_test = optimal_bayes_decision_with_threshold(datall, t)
            confus = valid.confusion_matrix(lab_test, labels_real)
            risk_test = bayes_detection_function_with_confusion(confus, posterior_prob, false_negative_cost,
                                                                false_positive_cost)
            res.append(risk_test / normalizer_factor)
        print("minDCF: ", min(res))


def ROC_plot():
    data = np.load("datasets/lab7/commedia_llr_infpar.npy")
    real_lab = np.load("datasets/lab7/commedia_labels_infpar.npy")
    thresh = np.linspace(min(data), max(data), 1000)
    res = []
    for t in thresh:
        lab_test = optimal_bayes_decision_with_threshold(data, t)
        confusion_matrix = valid.confusion_matrix(lab_test, real_lab)
        TPR = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
        FPR = confusion_matrix[0, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1])
        res.append((FPR, TPR))
    arr = np.array(res)
    plt.scatter(arr[:, 0], arr[:, 1])
    plt.show()


def bayes_error_plots():
    effPriorLogOdds = np.linspace(-3, 3, 21)
    pis = 1/(1+np.exp(effPriorLogOdds))
    data = np.load("datasets/lab7/commedia_llr_infpar.npy")
    real_lab = np.load("datasets/lab7/commedia_labels_infpar.npy")
    DCFlist = []
    minDCFlist = []
    for pi in pis:
        decision = optimal_bayes_decision_with_ratio(data, pi, 1, 1)
        confusion = valid.confusion_matrix(decision, real_lab)
        DCF = bayes_detection_function_with_confusion(confusion, pi, 1, 1)/min(pi, 1-pi)
        minDCF = minimal_detection_cost(data, real_lab, pi, 1, 1)
        DCFlist.append(DCF)
        minDCFlist.append(minDCF)
    plt.plot(effPriorLogOdds, DCFlist, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCFlist, label='minDCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()
    pass


def comparing_recognizers():
    data_1_labels = np.load("datasets/lab7/commedia_labels_infpar_eps1.npy")
    data_1 = np.load("datasets/lab7/commedia_llr_infpar_eps1.npy")
    settings = [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]
    for prior_prob, cfn, cfp in settings:
        decision = optimal_bayes_decision_with_ratio(data_1, prior_prob, cfn, cfp)
        confusion = valid.confusion_matrix(decision, data_1_labels)
        DCF = bayes_detection_function_with_confusion(confusion, prior_prob, cfn, cfp)/min(prior_prob*cfn, (1-prior_prob)*cfp)
        minDCF = minimal_detection_cost(data_1, data_1_labels, prior_prob, cfn, cfp)
        print("setting: ", prior_prob, cfn, cfp)
        print("DCF: ", DCF)
        print("minDCF: ", minDCF)


def multiclass_evaluation():
    multi_data = np.load("datasets/lab7/commedia_ll.npy")
    multi_labels = np.load("datasets/lab7/commedia_labels.npy")
    cost = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    pi = np.array([[0.3], [0.4], [0.3]])
    cost_matrix = cost @ np.exp(multi_data)
    calc_labels = np.argmin(cost_matrix, axis=0)
    norm_DCF = np.min(cost @ pi)
    confusion = valid.confusion_matrix(calc_labels, multi_labels)
    mis_class_ratio = confusion/np.sum(confusion, axis=1)
    DCF = np.sum(pi.T*mis_class_ratio)
    print("M: ", confusion)
    print("norm_DCF:", DCF/norm_DCF)
    print("DCF: ", DCF)


if __name__ == "__main__":
    multiclass_evaluation()
