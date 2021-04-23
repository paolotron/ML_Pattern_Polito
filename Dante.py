
import ml_p.word_processing as w
import numpy as np
from random import shuffle
from ml_p import Classifier


def load_divina():
    h = [line.strip().lower() for line in open("datasets/inferno.txt", encoding="ISO-8859-1").readlines()]
    pu = [line.strip().lower() for line in open("datasets/purgatorio.txt", encoding="ISO-8859-1").readlines()]
    pa = [line.strip().lower() for line in open("datasets/paradiso.txt", encoding="ISO-8859-1").readlines()]
    return h, pu, pa


if __name__ == "__main__":
    hell, purgatory, paradise = load_divina()
    split_perc = 0.75
    divina = [hell, purgatory, paradise]
    [shuffle(cantica) for cantica in divina]
    train, test = [cantica[:int(len(cantica)*split_perc)] for cantica in divina],\
                  [cantica[int(len(cantica)*split_perc):] for cantica in divina]
    eps = 0.001
    BOF = w.BagOfWord()
    BOF.fit(sum(train, []))
    dataset_train = BOF.transform(sum(train, []))
    labels_train = np.array(sum([[i] * len(train[i]) for i in range(3)], []))
    classifier = Classifier.MultiNomial()
    classifier.fit(dataset_train, labels_train)
    pred_lab = [classifier.predict(BOF.transform(test[i])) == i for i in range(3)]
    results = [sum(lis)/len(lis)for lis in pred_lab]
    print(f"Precision for Inferno: {results[0]}\n"
          f" Precision for Purgatorio: {results[1]}"
          f"\n Precision for Paradiso: {results[2]}")
