import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
#
# df_train = pd.read_csv("train.csv")
# print(df_train
#       )
from exceptions import ModelNoTrainYetError


class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None
        self._W= None
        self.learning_rate = None
        self.log_cost = []
        self.log_w = []

    @staticmethod
    def loss(h : np.array, y: np.array):
        """ Mean Square error between a prediction (h) and label (y) """
        sq_error = (h - y) ** 2
        n = len(y)
        return 1.0 / (2 * n) * sq_error.sum()

    def predict(self, X: np.array):
        """ From X data, make a prediction according to model weigths"""
        Y = np.dot(X, self._W)
        return Y

    def _gradient_desc_iteration(self, x: np.array, y: np.array):
        """

        :param x: Input data
        :param y: Target data (label)
        :return:
        """

        h = self.predict(x)
        gradient = 1/len(x) * np.dot(x.T, h - y)
        self._W = self._W - self.learning_rate * gradient

    def fit(self, X, y, n_iter=100000, lr=0.01, plotting=True):
        X = self._normalize(X)
        X = self._reshape(X)

        self._W = np.zeros(X.shape[1]) if len(X.shape) > 1 else np.zeros(1)
        self.learning_rate = lr
        for i in range(n_iter):
            if plotting:
                self.log_w.append(self._W.copy())
                prediction = self.predict(X)
                cost = self.loss(prediction, y)
                self.log_cost.append(cost.copy())
            self._gradient_desc_iteration(X, y)

        if plotting:
            plt.title('Loss against number of iteration')
            plt.xlabel('Iterations number')
            plt.ylabel('Loss')
            plt.plot(self.log_cost)
            plt.show()
        self.X = X
        self.y = y
        return self

    @staticmethod
    def _normalize(x: pd.Series):
        x = (x - x.mean()) / x.std()
        return x

    @staticmethod
    def _reshape(x: np.array):
        if len(x.shape) < 2:
            x = np.concatenate(([x.values], [np.ones(x.shape[0])]))
            x = x.T
        else:
            x = np.concatenate((x.values, np.ones((x.shape[0], 1))), axis=1)
        return x


