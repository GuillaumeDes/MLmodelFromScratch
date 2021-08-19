from math import sqrt
from typing import List


def euclidean_distance(row1: List[float], row2: List[float]) -> float:
    """
    Return the euclidian distance between 2 rows
    :return: the euclidian distance
    """
    assert len(row1) == len(row2), "Vectors do not have the same length"
    distance = sqrt(sum([(a - b) ** 2 for a, b in zip(row1, row2)]))
    return distance


def get_nearest_neighbors(train_set: List[List[float]], input_row: List[float], K: int) -> List[int]:
    """
    return the indices of the nearest neighbors
    :param train_set: list of train rows
    :param input_row: row to find the neighbors
    :param K: number of neighbors to consider
    :return: the indices of the K nearest neighbors
    """
    train_set = [(index, train_row) for index,train_row in zip(range(len(train_set)), train_set)]
    distances = [(train_row[0], euclidean_distance(input_row, train_row[1])) for train_row in train_set]
    distances.sort(key=lambda tup: tup[1])
    neighbors_indices = [distances[i][0] for i in range(K)]
    return neighbors_indices


def predict(train_set: List[List[float]], labels_set: List[int], input_row: List[float], K: int) -> int:
    """

    :param train_set:
    :param labels_set:
    :param input_row:
    :param K:
    :return: the predicted class of the input row
    """
    neighbors_indices = get_nearest_neighbors(train_set, input_row, K)
    neighbors_classes = [labels_set[indice] for indice in neighbors_indices]
    predicted_class = max(set(neighbors_classes), key=neighbors_classes.count)
    return predicted_class




print("ok")
