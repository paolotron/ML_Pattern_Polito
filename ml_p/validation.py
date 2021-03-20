from blueprints import *
import numpy as np


class TrainTest(Meter):

    def __init__(self, split, shuffle=False):
        self.split = split
        self.shuffle = shuffle

    def score(self, x, y):
        size = x.shape[0]
        train_size = size*self.split
        index = np.random.choice(size, train_size)
        if self.shuffle:
            np.random.shuffle(index)

        train_x = x[index, :]
        train_y = y[index]
        test_x = x[~index, :]
        test_y = y[~index]

        for step, hypers in zip(self.steps, self.hypers):
            current = step(**hypers)
            if isinstance(step, Pipe):
                train_x = current.fit_transform(train_x, train_y)
                test_x = current.transform(test_x, test_y)
            elif isinstance(step, Faucet):
                current.fit(train_x, train_y)
                predicted_y = current.predict()
                return sum(predicted_y & test_y)/len(test_y)


