# KNN Model Performance Evaluation

This project evaluates the performance of a K-Nearest Neighbors (KNN) model on a given dataset. The performance is evaluated across different values of K and various train-test splits. The results, including accuracy and confusion matrices, are saved in a PDF file.

## Files Included

- `data.csv`: The dataset used for training and testing the KNN model.
- `KNN_model_performance.pdf`: The PDF file containing the results of the evaluations.
- `knn_model_evaluation.ipynb`: The Jupyter notebook with the implementation of the KNN model evaluation.
- `README.md`: This file, providing an overview of the project.

## Project Overview

The goal of this project is to assess how the KNN algorithm performs with different values of K (number of neighbors) and various train-test splits. We aim to understand how the size of the training set and the choice of K affect the model's accuracy and the structure of the confusion matrix.

## Steps Involved

1. **Import Necessary Libraries**: Import the required Python libraries for data manipulation, model building, evaluation, and visualization.

2. **Load and Preprocess the Data**: Load the dataset from `data.csv` and preprocess it by handling missing values and normalizing the pixel values.

3. **Define Evaluation Function**: Define a function to train and evaluate the KNN model for given training and testing data and a specific value of K.

4. **Evaluate Model Across Different Scenarios**: Loop over different train-test splits and values of K to evaluate the KNN model's performance. Store the results in a list.

5. **Save Results in a PDF**: Create a PDF file and save the accuracy and confusion matrices for each scenario.

## Detailed Steps

### Step 1: Import Necessary Libraries

We start by importing the necessary libraries, including pandas for data manipulation, numpy for numerical operations, sklearn for machine learning tasks, matplotlib and seaborn for data visualization, and fpdf for generating PDF reports.

### Step 2: Load and Preprocess the Data

- **Loading Data**: Load the dataset from `data.csv`.
- **Handling Missing Values**: Check for and handle any missing values by filling them with zeros.
- **Separating Features and Labels**: Split the data into features (X) and labels (y).
- **Normalization**: Normalize the pixel values to ensure the model performs well with data of varying scales.

### Step 3: Define Evaluation Function

Define a function `evaluate_knn` that:
- Trains the KNN model with the specified number of neighbors (K).
- Makes predictions on the test set.
- Calculates the accuracy of the model.
- Generates the confusion matrix.

### Step 4: Evaluate Model Across Different Scenarios

Loop over different train-test splits (60:40, 70:30, 75:25, 80:20, 90:10, 95:5) and K values (2, 4, 5, 6, 7, 10):
- Split the data into training and testing sets based on the specified ratio.
- Train and evaluate the model for each combination of train-test split and K value.
- Store the results, including accuracy and confusion matrix, for each scenario.

### Step 5: Save Results in a PDF

Generate a PDF file using the `FPDF` library to document the performance of the KNN model:
- Include the train-test split ratio and the K value for each scenario.
- Present the accuracy and confusion matrix for each scenario.
- Visualize the confusion matrices using heatmaps for better understanding.

## Instructions

1. **Run the Jupyter Notebook**: Open and run the `knn_model_evaluation.ipynb` notebook to preprocess the data, evaluate the KNN model, and generate the results.

2. **Check the Results**: The results, including accuracy and confusion matrices for each train-test split and value of K, are saved in the `KNN_model_performance.pdf` file.

3. **Analyze the Results**: Use the results to analyze the dependency of the model's performance on the training-testing split and the value of K. Consider the following:
    - How does the accuracy change with different train-test splits?
    - Which value of K gives the best performance?
    - How do the confusion matrices vary across different scenarios?

## Conclusion

This project provides a comprehensive evaluation of the KNN model's performance on the given dataset. By systematically varying the train-test splits and the value of K, we can gain insights into the factors that influence the model's accuracy and predictive capability. The results documented in the PDF file serve as a valuable resource for understanding the behavior of the KNN algorithm in different settings.
