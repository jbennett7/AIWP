#!/usr/bin/python3

# Preprocessing data
# We deal with a lot of raw data in the real world. Machine learning algorithms expect
# data to be formatted in a certain way before they start the training process. In order
# the data for ingestion by machine learning algorithms, we have to preprocess it and
# convert it into the right format. Let's see how to do it.

import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
                      [-1.2, 7.8, -6.1],
                      [3.9, 0.4, 2.1],
                      [7.3, -9.9, -4.5]])

# Binarize data
# This process is used when we want to convert our numerical values into
# boolean values. In this example any value above 2.1 is 1 and everything
# else is 0
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBinarized data:\n", data_binarized)


# Mean Removal
# Removing the mean is a common preprocessing technique used in machine learning.
# It's usually useful to remove the mean from our feature vector, so that each
# feature is centered on zero. We do this in order to remove bias from the
# features in our feature vector.

# Print mean and standard deviation
print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Remove mean
data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# Scaling
# In our feature vector, the value of each feature can vary between many
# random values. So it becomes important to scale those features so that it is
# a level playing field for the machine learning algorithm to train on. We don't
# want any feature to be artifically large or small just because of the nature 
# of the measurements.


# Min max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

# Normalize data
# We use the process of normalization to modify the values
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)
