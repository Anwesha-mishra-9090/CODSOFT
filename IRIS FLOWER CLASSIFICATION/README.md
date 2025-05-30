# Iris Flower Classification - Advanced Machine Learning Model

This project implements an advanced machine learning pipeline to classify Iris flowers into their respective species (`setosa`, `versicolor`, `virginica`) based on sepal and petal measurements. The project includes comprehensive data loading, preprocessing, model training with hyperparameter tuning, evaluation, and feature importance visualization.

## Project Overview

The Iris flower dataset is a classic dataset for classification tasks. This project demonstrates how to build a robust classification model using the Random Forest algorithm with the following key features:

- Robust dataset loading with encoding fallbacks  
- Detailed data preprocessing and validation  
- Data visualization for exploratory analysis  
- Hyperparameter tuning using `RandomizedSearchCV`  
- Model evaluation using classification report and confusion matrix  
- Visualizing feature importances to understand model decisions

## Installation

Make sure you have Python 3.7+ installed. Install the required Python libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

Usage
Place the iris.csv dataset file in the same directory as the Python script.

Run : 
python main.py
The script will perform the following steps:
Load and preprocess the Iris dataset
Visualize feature distributions and class separability
Train a Random Forest classifier with hyperparameter tuning
Evaluate model performance and output detailed classification metrics
Visualize feature importances
Dataset
The input dataset file iris.csv should contain the following columns:

sepal length
sepal width
petal length
petal width
species (target variable with categories: setosa, versicolor, virginica)
Example rows:

sepal length

sepal width

petal length

petal width

species

5.1

3.5

1.4

0.2

setosa

7.0

3.2

4.7

1.4

versicolor

6.3

3.3

6.0

2.5

virginica

You can download the Iris dataset from the UCI Machine Learning Repository or use the one provided by scikit-learn.

Files
main.py - Main Python script containing the full pipeline.
iris.csv - Input dataset (not included in the repository, please add manually).
README.md - This documentation file.
License
This project is open source and free to use under the MIT License.

Contact
For questions or feedback, please contact: [mishra.anwesha143@gmail.com]

Enjoy exploring and classifying Iris flowers with advanced machine learning!
