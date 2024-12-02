# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:59:14 2024

@author: Aytekin
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
SGD = tf.keras.optimizers.SGD

# Load data
filename = 'data_perceptron.txt'
path = os.path.join(os.path.dirname(__file__), filename)
"""
Load the data using the numpy loadtxt function

"""
text = np.loadtxt(path)
print(text)


# Separate the data (x1, x2) from the labels
data = text[:, :2]  # Select columns 0 and 1 (features)
labels = text[:, 2]  # Select column 2 (labels)
labels = labels.reshape((text.shape[0], 1))  # Reshape labels into a 2D array

print("Data:")
print(data)
print("Labels:")
print(labels)


# Plot input data
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')
plt.show()



# Define a TensorFlow Keras model (Perceptron)
model = Sequential()
model.add(Dense(units=1, input_dim=2, activation='sigmoid'))  # Perceptron with sigmoid activation

# Compile the model
model.compile(optimizer=SGD(learning_rate=1.0), loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(data, labels, epochs=1000, verbose=1, batch_size=data.shape[0])

# Plot training progress (loss)
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('Number of epochs')
plt.ylabel('Training loss')
plt.title('Training loss progress')
plt.grid()
plt.show()