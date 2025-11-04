# RecycleMe: Waste Classification Using Deep Learning

## Project Overview

RecycleMe is a deep learning-based waste classification system that can categorize waste items into four different categories:

- B3 (Hazardous Waste)
- Organic
- Recycle
- Residu

This project uses MobileNetV2 architecture for efficient and accurate waste classification, making it suitable for potential mobile applications and real-world deployment.

## Technical Details

### Model Architecture

- Base Model: MobileNetV2 (pre-trained on ImageNet)
- Additional Layers:
  - Flatten Layer
  - Dense Layer (256 units) with ReLU activation
  - Dropout Layer (0.2)
  - Dense Layer (128 units) with ReLU activation
  - Dropout Layer (0.2)
  - Output Layer (4 units) with Softmax activation

### Data Processing

- Image Size: 224x224 pixels
- Data Augmentation Techniques:
  - Width shift (20%)
  - Height shift (20%)
  - Horizontal flip
  - Rotation (20 degrees)
  - Zoom (20%)
  - Shear (20%)

### Training Configuration

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Batch Size: 128
- Early Stopping: Monitoring validation accuracy with 5 epochs patience
- Model Checkpointing: Saves best model based on validation accuracy

## Dataset

The project uses a custom waste dataset obtained from Kaggle, containing images of different types of waste. The dataset is split into:

- Training Set (80% of total data)
- Validation Set (20% of total data)
- Separate Test Set for final evaluation

## Model Performance

The model is evaluated on various metrics including:

- Test Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix

## Files in the Project

- `Final_RecycleMe.ipynb`: Main Jupyter notebook containing the complete code
- `model_checkpoint.keras`: Best model checkpoint during training
- `Recycleme-Model.h5`: Final exported model in H5 format

## Requirements

- TensorFlow
- NumPy
- Pillow (PIL)
- Matplotlib
- Scikit-learn
- Scikit-image
- Kaggle API (for dataset download)

## Usage

1. Set up the Kaggle API credentials
2. Run the notebook to:
   - Download and prepare the dataset
   - Train the model
   - Evaluate the results
   - Export the model for deployment

## Future Improvements

- Implement real-time classification
- Create a web or mobile interface
- Expand the dataset with more waste categories
- Optimize model for mobile deployment

## Repository 

#### Project Capstone - RecycleMe Apps [[Link for Project Repository](http://github.com/Juli-Yash/RecycleMe)]

