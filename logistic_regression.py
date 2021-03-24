import numpy as np
from pandas import DataFrame, Series

from exceptions_custom import DataframeNotDefined, LabelOrFeatureMissing


class LogisticRegressionModel:
    """ Class model for monovariate logistic regression"""

    def __init__(self):
        """ Initalize weigths to 0"""
        self._weights = np.zeros([2])
        self.dataframe = None
        self.X = None
        self.Y = None
        self.lr = 0.01
        self.epochs = 1000

    def define_dataframe(self,
                         dataframe: DataFrame) -> None:
        """ Let the user define the data with a pandas dataframe"""
        self.dataframe = dataframe

    def define_variable_column(self,
                               x: str,
                               y: str) -> None:
        """ Let the user define which column to use for X and Y"""
        if ~self.dataframe.empty:
            self.X = self.dataframe[x].astype('float').values
            self.Y = self.dataframe[y].astype('float').values
        else:
            raise DataframeNotDefined("Please indicate a dataframe before choosing variable columns")

    def define_hyperparameters(self,
                               **kwargs) -> None:
        if kwargs.get("learning_rate") is None and kwargs.get("epochs") is None:
            raise Warning("you didn't define any parameter, please define 'learning_rate' or 'epochs'")
        if kwargs.get("learning_rate"):
            self.lr = kwargs.get("learning_rate")
        if kwargs.get("epochs"):
            self.epochs = kwargs.get("epochs")

    @staticmethod
    def _add_intercept(x: np.array) -> np.array:
        """ Add a column of one to numpy array X """
        intercept = np.ones((x.shape[0], 1))
        x = x.reshape(x.shape[0], 1) if len(x.shape) != 2 else x
        return np.concatenate((intercept, x), axis=1)

    def fit(self) -> np.array:
        """ Make the model fit to the data """
        if self.X is None or self.Y is None:
            raise LabelOrFeatureMissing("Please define a dataframe and select X and Y columns")

        x = self._add_intercept(self.X)
        weights = self._weights
        y = self.Y
        lr = self.lr

        for i in range(self.epochs):
            z = np.dot(x, weights)
            h = self._sigmoid(z)
            gradient = np.dot(x.T, (h - y)) / y.size
            weights -= lr * gradient
            if i % 1000 == 0:
                e = self._loss(h, y)
                print(f'loss: {e} \t')
        return weights

    @staticmethod
    def _sigmoid(z: any) -> any:
        """ The sigmoid function definition """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _loss(h: float,
              y: float) -> float:
        """ gap between prediction and gt : cross entropy loss"""
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def predict(self,
                serie: Series) -> Series:
        """ From a pandas Serie, make a prediction"""
        X = serie.to_numpy()
        X = self._add_intercept(X)
        Y_pred = self._sigmoid(np.dot(X, self._weights)).round()
        return Series(Y_pred, dtype=int, name="Y_pred")
