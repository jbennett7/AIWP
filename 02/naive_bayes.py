#!/usr/bin/python3

# Naive Bayes Classifier
# Naive Bayes is a technique used to build classifiers using Bayes theorem. Bayes theorem
# describes the probability of an event occurring based on different conditions that are
# related to this event. We build a Naive Bayes classifier by assigning class labels to
# problem instances. These problem instances are represented as vectors of feature values.
# The assumption here is that the value of any given feature is independent of the value
# of any other feature. This is called the independence assumption, which is the naive
# part of a Naive Bayes classifier.

# Given the class variable, we can just see how a given feature affects, it regardless
# of its affect on other features. For example, an animal may be considered a cheetah
# if it is spotted, has four legs, has a tail, and runs at about 70 MPH. A Naive Bayes
# classifier considers that each of these features contributes independently to the
# outcome. The outcome refers to the probability that this animal is a cheetah. We
# don't concern ourselves with the correlations that may exist between skin patterns,
# number of legs, presence of a tail, and movement speed. Let's see how to build a
# Naive Bayes classifier.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

from utilities import visualize_classifier

# Input file containing data
input_file = 'data_multivar_nb.txt'

# Load data from input file
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Create Naive Bayes classifier
classifier = GaussianNB()

# Train the classifier
classifier.fit(X, y)

# Predict the values for training data
y_pred = classifier.predict(X)

# Compute accuracy
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

# Visualize the performance of the classifier
visualize_classifier(classifier, X, y)

# Split data into traning and test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
    test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

# compute accuracy of the classifier
accuracy = 100.0 * (y_test == y_test_pred).sum() /X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy, 2), "%")

# Visualize the performance of the classifier
visualize_classifier(classifier_new, X_test, y_test)

num_folds = 3
accuracy_values = cross_validation.cross_val_score(classifier,
    X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_validation.cross_val_score(classifier,
    X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_validation.cross_val_score(classifier,
    X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_validation.cross_val_score(classifier,
    X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")
