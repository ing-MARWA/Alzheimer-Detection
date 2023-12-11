# Alzheimer-Detection
Creating a well-documented `README.md` file is essential for explaining your project to others. Here's a template tailored for your Alzheimer's disease detection project using MRI scans and a Streamlit app. You can adapt the text to add more specific details if needed.

## Alzheimer's Disease Detection Using CNNs Project

### Introduction

This repository contains the implementation of a Convolutional Neural Network (CNN) to classify MRI images into different stages of Alzheimer's disease, including Mild Demented, Moderate Demented, Non-Demented, and Very Mild Demented. The project aims to assist in the early detection and intervention of Alzheimer's through automated analysis of brain scans.

### Dataset

The data used for this project is sourced from a Kaggle dataset of Alzheimer's MRI images, which includes images segregated into the aforementioned stages. Each image has been pre-processed and normalized before being fed into the CNN for training. The dataset can be found [here](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset).

### Model Overview

The project utilizes a Sequential CNN architecture with convolutional layers, max-pooling, and fully connected layers, along with appropriate activation functions to effectively learn the features from the MRI images. For model training, the Adam optimizer and categorical cross-entropy loss function are used, with accuracy as the metric.

In addition, a Streamlit web application has been developed to facilitate easy use and demonstration of the model. Users can upload an MRI image and receive a prediction for Alzheimer's disease staging.

### Repository Structure

 - `Alzheimer Detection.ipynb`: upyter notebook contains the complete code for model training, evaluation, and visualization of results.
         - The notebook includes sections for :
         - Data preprocessing and loading
         - Model definition and training
         - Evaluation metrics calculation and visualization
         - Confusion matrix and classification report generation
         - Hyperparameter optimization (optional)
         - Model saving and loading
### Streamlit App
- The streamlit_app.py file contains the code for the Streamlit app.
- This app allows users to :
          - Upload an MRI image
          - Receive a prediction for Alzheimer's disease stage
          - View the predicted class and associated probability
          - Visualize the uploaded image
### Future Work : 
- Improve model performance by exploring different architectures and hyperparameter tuning.
- Include additional features like image segmentation and data augmentation.
- Integrate the model with medical imaging software for clinical use.
- Develop explainability methods to understand the model's decision-making process.
 ### Disclaimer
This project is intended for educational purposes only and should not be solely relied upon for medical diagnosis. Consult with healthcare professionals for definitive diagnosis and treatment plans.
### Installation
to run the project, follow these steps:  
    - Install required libraries:
        - pip install -r requirements.txt
        - Download the dataset and place it in the appropriate directory.
        - Run the Jupyter notebook:
             - jupyter notebook `Alzheimer Detection.ipynb`
       - Start the Streamlit app:
            - streamlit run streamlit_app.py
