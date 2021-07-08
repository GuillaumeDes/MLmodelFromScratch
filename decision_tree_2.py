import numpy as np
from collections import Counter
from typing import List, Union


def entropy(vector: List) -> float:
    """
    Application of entropy formula
    :param vector: vector representing the dataset labels
    :return: entropy
    """
    counts = np.bincount(np.array(vector, dtype=np.int64))
    percentages = counts / len(vector)

    entropy = sum([-p * np.log2(p) for p in percentages if p > 0])

    return entropy


def information_gain(parent: List, left_child: List, right_child: List) -> float:
    """
    Apply the gain formula for a given split
    :param parent: vector representing the dataset labels before split
    :param left_child: vector representing the left part of a split
    :param right_child: ector representing the right part of a split
    :return: gain
    """
    proportion_left = len(left_child) / len(parent)
    proportion_right = len(right_child) / len(parent)

    gain = entropy(parent) - (proportion_left * entropy(left_child) + proportion_right * entropy(right_child))
    return gain


def best_split(X: np.array, y: Union[np.array, List]):
    """
    find the best split among all the columns (=features) and all the thresholds (=each values in the column)
    :param X: features
    :param y: target
    :return:
    """
    best_split = {}
    best_info_gain = -1
    n_rows, n_cols = X.shape

    # for loop over each feature
    for feature in range(n_cols):
        X_feature = X[:, feature]
        # for loop over all unique value of the column, will be used as threshold to split the dataset
        for threshold in np.unique(X_feature):
            df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
            df_left = np.array([row for row in df if row[feature] <= threshold])
            df_right = np.array([row for row in df if row[feature] > threshold])


            if len(df_left) > 0 and len(df_right) > 0:

                # calculate only if not empty df right and left
                y = df[:, -1]
                y_left = df_left[:, -1]
                y_right = df_right[:, -1]

                info_gain = information_gain(y, y_left, y_right)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = {
                        'feature_index': feature,
                        'threshold': threshold,
                        'df_left': df_left,
                        'df_right': df_right,
                        'gain': info_gain
                    }
    return best_split


class Node:
    """
    Store the data about the feature, threshold, data going left and right, information gain and leaf node value
    """

    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value


class DecisionTree:
    """
        Decision tree classifier algorithm.
    """

    def __init__(self, min_samples_split=2, max_depth=5):
        """
        min_samples_split, max_depth are hyperparameters used for exit condition
        :param min_samples_split: minimum number of samples required to split a node
        :param max_depth: max depth of tree
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _build(self, X, y, depth=0):
        """
        recursive function to build the decision tree
        :param X:
        :param y:
        :param depth:
        :return:
        """
        n_rows, n_cols = X.shape

        # Is the node a leaf node ?
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            best = best_split(X, y)
            # If the split isn't pure
            if best['gain'] > 0:
                # build a new node with left split
                left = self._build(
                    X=best['df_left'][:, :-1],
                    y=best['df_left'][:, -1],
                    depth=depth + 1
                )
                # build a new node with right split
                right = self._build(
                    X=best['df_right'][:, :-1],
                    y=best['df_right'][:, -1],
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'],
                    threshold=best['threshold'],
                    data_left=left,
                    data_right=right,
                    gain=best['gain']
                )
        # the node is a leaf
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )

    def fit(self, X, y):
        '''
        Function used to train a decision tree classifier model.
        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        # Call a recursive function to build the tree. "the first "build", which will build root node, which will create
        # other nodes, etc...
        self.root = self._build(X, y)

    def _predict(self, x, tree):
        '''
        Helper recursive function, used to predict a single instance (tree traversal).
        :param x: single observation
        :param tree: built tree
        :return: float, predicted class
        '''
        # if the node is a Leaf node
        if tree.value != None:
            # return the value given by the leaf
            return tree.value

        # we only look the feature specified by the node
        feature_value = x[tree.feature]

        # Go to the left
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.data_left)

        # Go to the right
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.data_right)

    def predict(self, X):
        '''
        Function used to classify new instances.

        :param X: np.array, features
        :return: np.array, predicted classes
        '''
        # Call the _predict() function for every sample
        return [self._predict(x, self.root) for x in X]


