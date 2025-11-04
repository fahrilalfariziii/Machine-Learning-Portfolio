# Google Colab Projects Portfolio

This directory contains a collection of machine learning projects implemented using Google Colab. Each project demonstrates different aspects of artificial intelligence and machine learning techniques.

## Projects Overview

### 1. Brain Tumor Detection

Location: `Brain Tumor/brain_tumor_using_cnn.ipynb`

A Convolutional Neural Network (CNN) based system for detecting brain tumors in medical images.

**Technologies Used:**

- TensorFlow/Keras
- CNN Architecture
- Image Processing
- Binary Classification

**Key Features:**

- Custom CNN architecture with multiple convolutional and pooling layers
- Data augmentation techniques for improved model generalization
- Train/Test/Validation split for robust evaluation
- Performance visualization and metrics
- Binary classification (tumor/no tumor)

### 2. Population Prediction

Location: `population prediction/Untitled14.ipynb`

A machine learning model for predicting Indonesia's population growth using historical data.

**Technologies Used:**

- TensorFlow/Keras
- Scikit-learn
- K-Fold Cross Validation
- Time Series Analysis

**Key Features:**

- Neural Network regression model with dropout layers
- 10-fold Cross Validation for robust evaluation
- MinMaxScaler for data normalization
- Future population predictions (2024-2025)
- Performance metrics:
  - Mean Squared Error (MSE)
  - R² Score
- Visualization of training/validation loss and predictions

### 3. Rock Paper Scissors Classifier

Location: `Roshambo/detect_rock_paper_scissors.ipynb`

A computer vision project that classifies hand gestures into rock, paper, or scissors categories.

**Technologies Used:**

- TensorFlow/Keras
- CNN
- Image Data Augmentation
- Multi-class Classification

**Key Features:**

- Custom CNN architecture optimized for hand gesture recognition
- Real-time classification capability
- Advanced data augmentation:
  - Rotation
  - Width/Height shifts
  - Shear transformations
  - Zoom
  - Horizontal flip
- High accuracy target (>97%)
- Custom callback for training optimization

## Technical Implementation Details

### Common Technologies Across Projects:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Google Colab
- Jupyter Notebooks

### Best Practices Implemented:

- Data Preprocessing
- Model Validation
- Performance Metrics
- Visualization Techniques
- Early Stopping
- Data Augmentation

## Getting Started

To run any of these projects:

1. Open the desired `.ipynb` file in Google Colab
2. Ensure all required datasets are available
3. Run all cells in sequence
4. Follow the in-notebook documentation