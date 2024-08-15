# MNIST Neural Network Classification

This repository contains a Python script for training a neural network on the MNIST dataset using TensorFlow and Keras. The script preprocesses the data, builds a simple neural network model, trains the model, evaluates its performance, and saves the trained model to a file.

## Overview

The script performs the following tasks:

1. **Load and Preprocess Data**: 
   - Downloads the MNIST dataset.
   - Normalizes the pixel values of the images.
   - Converts class labels to one-hot encoded format.

2. **Visualize Data**:
   - Displays an example image from the dataset along with its label.

3. **Build Neural Network Model**:
   - Constructs a simple feedforward neural network using TensorFlow and Keras.
   - The model consists of:
     - A Flatten layer to reshape the input.
     - A Dense hidden layer with 5 neurons and ReLU activation.
     - A Dense output layer with 10 neurons (one for each class) and softmax activation.

4. **Compile and Train Model**:
   - Compiles the model with the Adam optimizer and categorical crossentropy loss function.
   - Trains the model on the training data for 5 epochs with a batch size of 32.

5. **Evaluate Model**:
   - Evaluates the trained model on the test data and prints the performance metrics.

6. **Save Model**:
   - Saves the trained model to a file named `mnist.h5`.

## Requirements

- TensorFlow
- Keras
- Matplotlib

You can install the required packages using pip:

```bash
pip install tensorflow matplotlib
