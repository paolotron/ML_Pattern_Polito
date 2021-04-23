import numpy as np

from .blueprints import Pipe


class BagOfWord:

    def __init__(self, measure="occurances"):
        self.all_words = set()
        self.words_index = None
        self.measure = {
            "occurances": lambda i: i
        }[measure]

    def fit(self, x):
        """
        :param x: list of documents
        :return:
        """
        [self.all_words.update(document.strip().lower().split(" ")) for document in x]
        self.words_index = {w: i for i, w in enumerate(self.all_words)}

    def transform(self, x, compress=False):
        arr_list = []
        tot = np.zeros(len(self.words_index))
        for document in x:
            d = document.strip().lower().split(" ")
            a = np.zeros(len(self.words_index))
            for word in d:
                if word in self.words_index:
                    if not compress:
                        a[self.words_index[word]] += 1
                    else:
                        tot[self.words_index[word]] += 1
            arr_list.append(a)
        res = np.vstack(arr_list) if not compress else tot.reshape(1, -1)
        return self.measure(res)

    def fit_transform(self, x, compress=False):
        self.fit(x)
        return self.transform(x, compress)
