# -*- coding: utf-8 -*-
"""
Created on Mon Dec 02 13:31:54 2024

@author: Aytekin
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
SGD = tf.keras.optimizers.SGD

filename = 'data_simple_nn.txt'
path = os.path.join(os.path.dirname(__file__),filename) 
text = np.loadtxt(path)
print(text)

# separate the data
data = text[:, 0:2]
labels = text[:, 2:]

# Plot the data
# Plot input data
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# Normalize data to [0, 1]
data_min = data.min(axis=0)  # Minimum of each feature (column)
data_max = data.max(axis=0)  # Maximum of each feature (column)

print("data_min:", data_min)
print("data_max:", data_max)

data_normalized = (data - data_min) / (data_max - data_min)  # Apply Min-Max Scaling
print("data_normalized:")
print(data_normalized)


num_output = labels.shape[1] # Number of output neurons
model = Sequential()
# Add a single layer into the model
model.add(Dense(units=num_output, input_dim=2, activation='sigmoid'))

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.003), loss='mean_squared_error', metrics=['accuracy'])

# Train the neural network
history = model.fit(data_normalized, labels, epochs=1000, batch_size=data.shape[0], verbose=1)


 #Plot the training progress
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()

# Run the classifier on test datapoints
print('\nTest results:')
data_test = np.array([[0.4, 4.3], [4.4, 0.6], [4.7, 8.1], [0.9,7.4],[7,4],[4,7],[7.2,4.1]])
predictions = model.predict(data_test)

for item, prediction in zip(data_test, predictions):
    print(f"{item} --> {np.round(prediction, decimals=2)}")