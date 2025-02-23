# Handwritten-Digit-Classification-using-CNN

## Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras and trained on grayscale images of digits (0-9). The dataset consists of 60,000 training images and 10,000 test images.

## Dataset

The MNIST dataset is a collection of 28x28 pixel grayscale images of handwritten digits. It is widely used for benchmarking machine learning models in image classification tasks.

## Model Architecture

The CNN model consists of the following layers:

- Conv2D (32 filters, 3x3 kernel, ReLU activation): Extracts features from input images.

- MaxPooling2D (2x2 pool size): Reduces dimensionality and retains important features.

- Conv2D (64 filters, 3x3 kernel, ReLU activation): Extracts more complex features.

- MaxPooling2D (2x2 pool size): Further reduces dimensions.

- Flatten: Converts 2D feature maps into a 1D vector.

- Dense (128 neurons, ReLU activation): Fully connected layer for classification.

- Dropout (0.5): Prevents overfitting.

- Dense (10 neurons, softmax activation): Output layer for classification into 10 categories.

## Results

The model achieves an accuracy of approximately 95% on the MNIST test set.



