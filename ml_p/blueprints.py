from abc import ABC, abstractmethod


class NoFitError(Exception):
    def __init__(self, message="Fit the data before transforming"):
        super().__init__(message)


class Pipe(ABC):
    """
    Pipeline Step, The input is a Data Matrix and the output is
    a Data Matrix
    """

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def fit_transform(self, x, y):
        pass

    @abstractmethod
    def transform(self, x):
        pass


class Faucet(ABC):
    """
    General Data ML algorithms, classificators, regressors or clusterers
    """

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def fit_predict(self, x, y):
        pass


class Meter(ABC):
    """
    General Validation step
    """
    steps = None
    hypers = None

    def set_system(self, steps, hypers):
        if len(steps) != len(hypers):
            raise ValueError("Number of steps and hyperparameters should match")
        self.steps = steps
        self.hypers = hypers

    @abstractmethod
    def score(self, x, y):
        pass
