from unittest import TestCase
from KNN import euclidean_distance, get_nearest_neighbors, predict


class KNNTest(TestCase):
    def test_euclidian_distance(self):
        dataset = [[2.7810836, 2.550537003],
                   [1.465489372, 2.362125076],
                   [3.396561688, 4.400293529],
                   [1.38807019, 1.850220317],
                   [3.06407232, 3.005305973],
                   [7.627531214, 2.759262235],
                   [5.332441248, 2.088626775],
                   [6.922596716, 1.77106367],
                   [8.675418651, -0.242068655],
                   [7.673756466, 3.508563011]]
        row0 = dataset[0]
        computed_distance_list = [euclidean_distance(row0, row) for row in dataset]
        expected_distance_list = [
            0.0,
            1.3290173915275787,
            1.9494646655653247,
            1.5591439385540549,
            0.5356280721938492,
            4.850940186986411,
            2.592833759950511,
            4.214227042632867,
            6.522409988228337,
            4.985585382449795
        ]
        [self.assertAlmostEqual(expected_value, computed_value, 3) for expected_value, computed_value in
         zip(expected_distance_list, computed_distance_list)]

    def test_get_neighbors(self):
        dataset = [[2.7810836, 2.550537003],
                   [1.465489372, 2.362125076],
                   [3.396561688, 4.400293529],
                   [1.38807019, 1.850220317],
                   [3.06407232, 3.005305973],
                   [7.627531214, 2.759262235],
                   [5.332441248, 2.088626775],
                   [6.922596716, 1.77106367],
                   [8.675418651, -0.242068655],
                   [7.673756466, 3.508563011]]

        computed_neighbors = get_nearest_neighbors(dataset, dataset[0], 3)
        expected_neighbors = [0, 4, 1]
        self.assertListEqual(expected_neighbors, computed_neighbors)

    def test_predict(self):
        dataset = [[2.7810836, 2.550537003],
                   [1.465489372, 2.362125076],
                   [3.396561688, 4.400293529],
                   [1.38807019, 1.850220317],
                   [3.06407232, 3.005305973],
                   [7.627531214, 2.759262235],
                   [5.332441248, 2.088626775],
                   [6.922596716, 1.77106367],
                   [8.675418651, -0.242068655],
                   [7.673756466, 3.508563011]]

        classes = [0,0,0,0,1,1,1,1]
        computed_prediction = predict(dataset, classes, dataset[0], 3)
        expected_prediction = 0
        self.assertEqual(expected_prediction, computed_prediction)