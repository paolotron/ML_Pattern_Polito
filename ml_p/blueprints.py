from abc import ABC, abstractmethod


class NoFitError(Exception):
    def __init__(self, message="Fit the data before transforming"):
        super().__init__(message)


class Pipe(ABC):

    @abstractmethod
    def fit(self, x, labels):
        pass

    @abstractmethod
    def fit_transform(self, x, y):
        pass

    @abstractmethod
    def transform(self, x):
        pass
