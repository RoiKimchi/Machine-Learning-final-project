Got it! I'll create a detailed README that focuses on each task individually, covering the preprocessing steps, models used, and any specific techniques applied for each dataset. Let me draft it up for you:

---

# Machine Learning Final Course Project

This repository contains the final project for the Machine Learning course, focusing on three distinct machine learning tasks:

1. **Asteroid Hazard Classification**: A classification task to determine if an asteroid is hazardous.
2. **House Price Prediction in Poland**: A regression analysis predicting house prices based on recent data.
3. **Movie Recommendation System**: A recommendation system utilizing Neural Networks (NN) and Convolutional Neural Networks (CNN) with enhancements.

## Table of Contents

- [Project Overview](#project-overview)
- [Asteroid Hazard Classification](#asteroid-hazard-classification)
  - [Dataset Description](#dataset-description)
  - [Preprocessing](#preprocessing)
  - [Modeling](#modeling)
  - [Results](#results)
- [House Price Prediction in Poland](#house-price-prediction-in-poland)
  - [Dataset Description](#dataset-description-1)
  - [Preprocessing](#preprocessing-1)
  - [Modeling](#modeling-1)
  - [Results](#results-1)
- [Movie Recommendation System](#movie-recommendation-system)
  - [Dataset Description](#dataset-description-2)
  - [Preprocessing](#preprocessing-2)
  - [Modeling](#modeling-2)
  - [Results](#results-2)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

This project explores three different machine learning applications, each with its unique dataset and challenge. Below is a detailed breakdown of the processes, models, and results for each task.

## Asteroid Hazard Classification

### Dataset Description
The asteroid dataset includes information on various asteroids, such as their size, velocity, and distance from Earth. The goal was to classify whether an asteroid is hazardous or non-hazardous based on these features.

### Preprocessing
1. **Data Cleaning**: Handled missing values by imputing with median values and removed any duplicates.
2. **Feature Scaling**: Applied StandardScaler to normalize features like size and velocity.
3. **Feature Selection**: Used feature importance scores from a Random Forest model to select the most relevant features for classification.

### Modeling
- **Logistic Regression**: Initially used as a baseline model due to its simplicity and interpretability.
- **Random Forest Classifier**: Implemented to handle the non-linear relationships between features.
- **Support Vector Machine (SVM)**: Applied with a radial basis function (RBF) kernel to maximize classification accuracy.

### Results
- **Accuracy**: Achieved an accuracy of 92% on the test set with the Random Forest Classifier.
- **Key Insights**: The asteroid's size and distance from Earth were the most significant predictors of hazard classification.

## House Price Prediction in Poland

### Dataset Description
This dataset comprises house prices in various cities across Poland, collected in June 2024. The objective was to predict house prices based on features such as location, number of rooms, and square footage.

### Preprocessing
1. **Data Cleaning**: Removed outliers that significantly deviated from the mean price.
2. **Encoding**: Used one-hot encoding for categorical variables like location and type of housing.
3. **Feature Engineering**: Created new features such as price per square meter to improve model performance.

### Modeling
- **Linear Regression**: Served as a baseline model to understand the relationship between features and house prices.
- **XGBoost Regressor**: Applied to capture complex patterns in the data with its boosting approach.
- **Random Forest Regressor**: Used for its robustness against overfitting, especially with noisy data.

### Results
- **R^2 Score**: Achieved an R^2 score of 0.85, indicating a strong correlation between predicted and actual prices.
- **Key Insights**: Location and the number of rooms were the most influential factors in predicting house prices.

## Movie Recommendation System

### Dataset Description
The movie dataset includes user ratings, movie genres, and other metadata. The task was to build a recommendation system that suggests movies based on user preferences.

### Preprocessing
1. **Data Cleaning**: Removed movies with very few ratings to avoid skewed recommendations.
2. **Normalization**: Applied MinMaxScaler to normalize rating scores between 0 and 1.
3. **Train-Test Split**: Divided the dataset into training and testing sets, ensuring a balanced distribution of ratings.

### Modeling
- **Neural Network (NN)**: Implemented a basic NN architecture to learn user-movie interaction patterns.
- **Convolutional Neural Network (CNN)**: Enhanced the model by using CNN layers to capture intricate patterns in user behavior.
- **Hybrid Model**: Combined the NN and CNN approaches to further improve recommendation accuracy.

### Results
- **Precision/Recall**: The hybrid model achieved a precision of 0.78 and recall of 0.74 on the test data.
- **Key Insights**: The CNN layers significantly improved the model's ability to recommend less popular movies that are highly rated by niche audiences.

## Conclusion

This project demonstrates the application of machine learning techniques across various domains. Each task provided unique challenges and insights, contributing to a comprehensive understanding of classification, regression, and recommendation systems.
