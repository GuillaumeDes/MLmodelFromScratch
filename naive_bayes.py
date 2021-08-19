from typing import List, Dict, Tuple
from statistics import mean, stdev
from math import exp, sqrt, pi


def separate_by_class(dataset: List[List[float]], labels: List[int]) -> Dict:
    """ Step 1: , calculate the probability of data by the class they belong to, the so-called base rate."""
    separated = dict()
    for sample, label in zip(dataset, labels):
        if (label not in separated):
            separated[label] = list()
        separated[label].append(sample)
    return separated


def summarize_dataset(dataset: List[List[float]]) -> List[Tuple]:
    """ Compute the mean, the standard deviation and the length of each column"""
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    return summaries


def summarize_by_class(dataset: List[List[float]], labels: List[int]) -> Dict:
    separated = separate_by_class(dataset, labels)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities
