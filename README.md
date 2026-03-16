Railway Complaint Image Classifier
Project Overview

This project builds a deep learning model that classifies railway complaint images into different categories. The system helps automate the process of identifying and categorizing complaints based on images submitted by users.

The goal is to assist railway authorities in quickly identifying issues and improving response time.

Problem Statement

Railway authorities receive numerous complaints from passengers in the form of images. Manually categorizing these complaints is time-consuming and inefficient.

This project uses a deep learning model to automatically classify complaint images into predefined categories.

Project Structure
complaint-classifier
│
├── src/
│   ├── preprocessing/
│   │   ├── cleanup_dataset.py
│   │   └── remove_corrupted_image.py
│   │
│   ├── training/
│   │   └── train_model.py
│   │
│   └── deployment/
│       ├── app.py
│       └── templates/
│           └── index.html
│
├── data/                 # Dataset (not included in repo)
├── models/               # Trained model files (ignored in Git)
├── prep.py               # Dataset preparation script
├── requirements.txt
└── README.md
Technologies Used

Python

TensorFlow / Keras

NumPy

OpenCV

Flask (for deployment)

HTML (for web interface)

Dataset

The dataset consists of railway complaint images belonging to multiple categories.

Note: The dataset is not included in the repository due to size limitations.

You can place the dataset inside the data/ folder before training the model.

Model Training

The training pipeline performs the following steps:

Dataset cleaning and preprocessing

Removal of corrupted images

Image resizing and normalization

Model training using a convolutional neural network (CNN)

Saving the trained model

Training script:

src/training/train_model.py
Deployment

A simple web interface is created using Flask to allow users to upload complaint images and get predictions.

Run the application:

python src/deployment/app.py

Then open:

http://localhost:5000
How to Run the Project
1. Clone the repository
git clone https://github.com/AoaAshI/complaint-classifier.git
2. Install dependencies
pip install -r requirements.txt
3. Train the model
python src/training/train_model.py
4. Run the web application
python src/deployment/app.py
Future Improvements

Improve model accuracy with transfer learning

Add more complaint categories

Deploy the application to a cloud platform

Build a full complaint management system

Author

Kushagra Khajuria

Data Science Enthusiast
