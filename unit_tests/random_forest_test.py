import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from random_forest import RandomForest


class TestRandomForest(unittest.TestCase):
    def test_random_forest(self):
        iris = load_iris()

        X = iris['data']
        y = iris['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForest(num_trees=5, max_depth=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        score = accuracy_score(y_test, preds)
        self.assertGreater(score, 0.9)