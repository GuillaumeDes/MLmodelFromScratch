from unittest import TestCase
from naive_bayes import separate_by_class, summarize_dataset, summarize_by_class, calculate_probability, \
    calculate_class_probabilities

__dataset_test__ = [[3.393533211, 2.331273381],
                    [3.110073483, 1.781539638],
                    [1.343808831, 3.368360954],
                    [3.582294042, 4.67917911],
                    [2.280362439, 2.866990263],
                    [7.423436942, 4.696522875],
                    [5.745051997, 3.533989803],
                    [9.172168622, 2.511101045],
                    [7.792783481, 3.424088941],
                    [7.939820817, 0.791637231]]

__labels_test__ = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


class NaiveBayesTest(TestCase):
    def test_separate_by_class(self):
        computed_separation = separate_by_class(__dataset_test__, __labels_test__)
        expected_separation = {0: [[3.393533211, 2.331273381], [3.110073483, 1.781539638], [1.343808831, 3.368360954],
                                   [3.582294042, 4.67917911], [2.280362439, 2.866990263]],
                               1: [[7.423436942, 4.696522875], [5.745051997, 3.533989803], [9.172168622, 2.511101045],
                                   [7.792783481, 3.424088941], [7.939820817, 0.791637231]]}
        self.assertListEqual(computed_separation[0], expected_separation[0])
        self.assertListEqual(computed_separation[1], expected_separation[1])

    def test_summarize_dataset(self):
        computed_summary = summarize_dataset(__dataset_test__)
        expected_sumary = [(5.1783333865, 2.7665845055177263, 10), (2.9984683241, 1.218556343617447, 10)]
        self.assertTupleEqual(computed_summary[0], expected_sumary[0])
        self.assertTupleEqual(computed_summary[1], expected_sumary[1])

    def test_summarize_by_class(self):
        computed_class_summary = summarize_by_class(__dataset_test__, __labels_test__)
        expected_class_summary = {0: [(2.7420144012, 0.9265683289298018, 5), (3.0054686692, 1.1073295894898725, 5)],
                                  1: [(7.6146523718, 1.2344321550313704, 5),
                                      (2.9914679790000003, 1.4541931384601618, 5)]}
        self.assertListEqual(computed_class_summary[0], expected_class_summary[0])
        self.assertListEqual(computed_class_summary[1], expected_class_summary[1])

    def test_calculate_probability(self):
        self.assertAlmostEqual(0.3989, calculate_probability(1, 1, 1), 3)

    def test_calculate_class_probabilities(self):
        computed_class_summary = summarize_by_class(__dataset_test__, __labels_test__)
        computed_probability_dict = calculate_class_probabilities(computed_class_summary, __dataset_test__[0])
        expected_probability_dict = {0: 0.05032427673372075, 1: 0.00011557718379945765}
        self.assertAlmostEqual(computed_probability_dict[0], expected_probability_dict[0], 5)
        self.assertAlmostEqual(computed_probability_dict[1], expected_probability_dict[1], 5)
