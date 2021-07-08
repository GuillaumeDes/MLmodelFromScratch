import unittest
from decision_tree_2 import entropy, information_gain, DecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class TestRegressionTree(unittest.TestCase):
    def test_entropy(self):
        vector = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        expected_entropy = 0.88129
        computed_entropy = entropy(vector)
        self.assertAlmostEqual(computed_entropy, expected_entropy, 5)

    def test_information_gain(self):
        parent = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        left_child = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        right_child = [0, 0, 0, 0, 1, 1, 1, 1]
        expected_gain = 0.18094
        computed_gain = information_gain(parent, left_child, right_child)
        self.assertAlmostEqual(computed_gain, expected_gain, 5)

    def test_decision_tree(self):
        iris = load_iris()

        X = iris['data']
        y = iris['target']

        # Split dataset in a train dataset and a test dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DecisionTree(min_samples_split=2, max_depth=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.assertListEqual(y_test.tolist(), y_pred)
