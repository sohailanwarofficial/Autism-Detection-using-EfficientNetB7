# Autism Detection using EfficientNetB7

This project is a deep learning-based solution for detecting autism in medical images. It utilizes the powerful EfficientNetB7 convolutional neural network (CNN) architecture to perform binary classification, distinguishing between autistic and non-autistic subjects based on medical images.

## Overview

The code provided in this repository includes:

- Data preprocessing using TensorFlow's `ImageDataGenerator` for augmenting and loading training, test, and validation data.
- Transfer learning with the pre-trained EfficientNetB7 model.
- Fine-tuning the model for binary classification.
- Training the model with customizable hyperparameters.
- Plotting training and validation accuracy and loss graphs.
- Evaluating the model's performance on test data.
- Making predictions on individual test images.

## Getting Started

Before running the code, make sure to set up your environment and dataset. Here are the initial steps:

1. Import necessary libraries and mount Google Drive (if using Google Colab).
2. Define paths and constants for your dataset, including `train_path`, `test_path`, `valid_path`, `batch_size`, and `image_size`.
3. Create `ImageDataGenerator` instances for data augmentation and loading.
4. Load and preprocess the training, test, and validation data using the generators.

## Pre-trained Model

The code leverages the pre-trained EfficientNetB7 model for feature extraction. It loads the model, freezes its layers, and adds custom classification layers to the end.

## Training

You can specify the number of epochs for training by adjusting the `epochs` variable. The model will be trained using the specified hyperparameters and will display training and validation accuracy and loss graphs.

## Model Evaluation

After training, the model's performance is evaluated on the test dataset, and the test accuracy is printed.

## Predictions

The code also includes functionality to make predictions on individual test images. It randomly selects a test image, displays it along with the true label, and shows the model's predicted probability.

## Saving the Model

The trained model can be saved to a specified path using the `model.save()` function.

Please customize the code as needed for your specific dataset and requirements. Feel free to modify the architecture, hyperparameters, and data preprocessing to achieve the best results for your autism detection task.

For any questions or issues, please don't hesitate to reach out.

**Note**: Ensure that you have the required libraries installed and the necessary data paths set correctly before running the code.
