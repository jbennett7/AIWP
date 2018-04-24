#!/usr/bin/python3

# Logistic Regression classifier
# Logistic regression is a technique that is used to explain the relationship between
# input variables and output variables. The input variables are assumed to be independent
# and the output variable is referred to as the dependent variable. The dependent variable
# can take only a fixed set of values. These values correspond to the classes of the
# classification problem.

# Our goal is to identify the relationship between the independent variables and the
# dependent variables by estimating the probabilities using a logistic function. This
# logistic function is a sigmoid curve that's used to build the functionw ith various
# paramters. It is very closely related to generalized linear model analysis, where we
# try to fit a line to a bunch of points to minimize the error. Instead of using linear
# regression, we use logistic regression. Logistic regression by itself is actually not
# a classification technique, but we use it in this way so as to facilitate classification.
# It is used very commonly  in machine learning because of its simplicity.
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from utilities import visualize_classifier

# Define sample input data
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5],
    [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# Create the logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)

# Train the classifier
classifier.fit(X, y)

# Visualize the performance of the classifier
visualize_classifier(classifier, X, y)
