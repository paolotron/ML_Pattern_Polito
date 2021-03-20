import numpy as np
import matplotlib.pyplot as plt
from ml_p.preproc import Pca
from ml_p.preproc import Lda
from ml_p.preproc import StandardScaler


def load(path):
    l = []
    matrix = []
    for line in open(path):
        arr = line.strip().split(',')
        l.append(arr[-1])
        nums = np.array(list(map(lambda i: float(i), arr[:-1])))
        matrix.append(nums)
    d = np.vstack(matrix)
    l = np.array(l)
    return d, l


def plot_all_iris():
    data, label = load("dataset/iris.csv")
    names = ('Sepal length', ' Sepal Width', 'Petal length', 'Petal width')
    for name, column in zip(names, data.T):
        fig, axes = plt.subplots()
        for lab in set(label):
            mask = label == lab
            axes.hist(column[mask], alpha=0.6, label=lab, density=True, bins=8)
        axes.legend(set(label))
        axes.set_xlabel(name)
        fig.show()

    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            for lab in set(label):
                ax[i, j].legend(lab)
                if i == j:
                    ax[i, j].hist(data[:, i][label == lab])
                else:
                    ax[i, j].scatter(data[:, i][label == lab], data[:, j][label == lab])
    fig.show()


def plot(data, label):
    plist = []
    llist = []
    for lab in set(label):
        d = data[label == lab]
        r = plt.scatter(d[:, 0], d[:, 1])
        plist.append(r)
        llist.append(lab)
    plt.legend(plist, llist)
    plt.show()


def test_pca():
    data, label = load("datasets/iris.csv")
    comp = Pca(2).fit_transform(data)
    plot(comp, label)


def test_lda():
    data, label = load("datasets/iris.csv")
    comp = Lda(2).fit_transform(data, label)
    plot(comp, label)


def plot_reduction():
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    data, label = load("datasets/iris.csv")
    pc = Pca(2, True).fit_transform(data)
    ld = Lda(2, True).fit_transform(data, label)
    pc_ld = Lda(2, True).fit_transform(pc, label)
    for lab in set(label):
        d = data[label == lab]
        d_pc = pc[label == lab]
        d_ld = ld[label == lab]
        d_pc_ld = pc_ld[label == lab]

        ax[0, 0].scatter(d[:, 0], d[:, 1])
        ax[0, 1].scatter(d_pc[:, 0], d_pc[:, 1])
        ax[1, 0].scatter(d_ld[:, 0], d_ld[:, 1])
        ax[1, 1].scatter(d_pc_ld[:, 0], d_pc_ld[:, 1])

    ax[0, 0].title.set_text("DATA")
    ax[0, 1].title.set_text("PCA")
    ax[1, 0].title.set_text("LDA")
    ax[1, 1].title.set_text("PCA+LDA")
    fig.show()


if __name__ == '__main__':
    # plot_reduction()
    iris = load("datasets/iris.csv")
    plot(iris[0], iris[1])
    plot(StandardScaler().fit_transform(iris[0]), iris[1])
