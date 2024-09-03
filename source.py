import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Loading and splitting data into training data and testing data
mnist = tf.keras.datasets.mnist  # Loading the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Splitting the data into training and testing data

# Normalizing the data (pre-processing)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Building the model
model = tf.keras.models.Sequential()  # Basic sequential model
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Flatten the input, transforms the input of 28x28 pixels into a single array of 784 pixels
model.add(tf.keras.layers.Dense(128, activation="relu"))  # Dense layer with 128 neurons, activation function is ReLU
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))  # Output layer with 10 neurons, activation function is Softmax

# Compiling the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Fitting the model
model.fit(x_train, y_train, epochs=20)

# Saving the model in Keras format
model.save("digits.keras")

# Load the model without compiling
model = tf.keras.models.load_model("digits.keras", compile=False)

# Recompile the model with the necessary optimizer
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Evaluating the model through loss and accuracy values
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Testing and reading the image
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try: 
       image = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)
       image = cv2.resize(image, (28, 28))
       image = np.invert(np.array([image]))
       image = tf.keras.utils.normalize(image, axis=1)
       
       # Reshape image to add a batch dimension
       image = image.reshape(-1, 28, 28)
       
       prediction = model.predict(image)
       print(f"Prediction: {np.argmax(prediction)}")
       plt.imshow(image[0], cmap=plt.cm.binary)
       plt.show()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        image_number += 1
