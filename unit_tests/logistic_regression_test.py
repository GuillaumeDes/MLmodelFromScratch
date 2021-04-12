import unittest
import pandas as pd
import numpy as np
from logistic_regression import LogisticRegressionModel


class TestLogisticRegression(unittest.TestCase):
    """ This class includes all methods to test logistic regression class """

    def test_model_fitting(self):
        """ Declare a Logistic regression model, make it fit data and check the weights"""
        input_dataframe = pd.DataFrame({
            "X": pd.Series([50, 10, 20, 5, 95, 70, 100, 200, 0], dtype=int),
            "Y":  pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1], dtype=int),
        })
        x_column = "X"
        y_column = "Y"

        model = LogisticRegressionModel()
        model.define_dataframe(input_dataframe)
        model.define_variable_column(x_column, y_column)
        model.define_hyperparameters(learning_rate=0.01, epochs=10000)

        model.fit()

        expected_weights = np.array([4.6918633, -0.14748218])
        non_equality_message = "After fitting, weights are different from expected weights"
        np.testing.assert_array_almost_equal(expected_weights,
                                             model._weights,
                                             decimal=6,
                                             err_msg=non_equality_message)

    def test_model_predict(self):
        """ Declare a Logistic Regression model, manually write the weights and make a prediction"""
        input_dataframe = pd.DataFrame({
            "X": pd.Series([50, 10, 20, 5, 95, 70, 100, 200, 0], dtype=int),
            "Y":  pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1], dtype=int),
        })

        model = LogisticRegressionModel()
        model_input = input_dataframe["X"]
        model._weights = np.array([4.6918633, -0.14748218])
        prediction = model.predict(model_input)

        expected_prediction = input_dataframe["Y"]
        pd.testing.assert_series_equal(expected_prediction, prediction, check_names=False)

