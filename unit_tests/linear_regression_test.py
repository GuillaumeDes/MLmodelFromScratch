import unittest
import numpy as np
import pandas as pd
from linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lr = LinearRegression()

    def test_zero_h_zero_y(self):
        self.assertAlmostEqual(self.lr.loss(h=np.array([0, 0]), y=np.array([0, 1])), 0.25)

    def test_one_h_zero_y(self):
        self.assertAlmostEqual(self.lr.loss(h=np.array([1]), y=np.array([0])), 0.5)

    def test_two_h_zero_y(self):
        self.assertAlmostEqual(self.lr.loss(h=np.array([2]), y=np.array([0])), 2)

    def test_zero_h_one_y(self):
        self.assertAlmostEqual(self.lr.loss(h=np.array([0]), y=np.array([1])), 0.5)

    def test_zero_h_two_y(self):
        self.assertAlmostEqual(self.lr.loss(h=np.array([0, 0]), y=np.array([0, 3])), 2.25)

    def test_find_coefficients(self):
        df = pd.read_csv("train.csv")
        x = df['GrLivArea']
        y = df['SalePrice']


        self.lr.fit(x, y, n_iter=2000, lr=0.01)
        np.testing.assert_array_almost_equal(
            self.lr._W,
            np.array([56294.90199925, 180921.19555322])
        )

    def test_find_coefficients_multivariate(self):
        df = pd.read_csv("train.csv")
        x = df[['GrLivArea', "OverallQual"]]
        y = df['SalePrice']


        self.lr.fit(x, y, n_iter=2000, lr=0.01)
        np.testing.assert_array_almost_equal(
            self.lr._W,
            np.array([29356.822686,  45427.800376, 180921.195553])
        )

    def test_find_coefficients_multivariate_2(self):
        df = pd.read_csv("train.csv")
        x = df[['GrLivArea', "OverallQual", "GarageCars"]]
        y = df['SalePrice']


        self.lr.fit(x, y, n_iter=2000, lr=0.01)
        np.testing.assert_array_almost_equal(
            self.lr._W,
            np.array([[26631.938306,  37478.604254,  15921.225813, 180921.195553]])
        )