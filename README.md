# Steel Plates Quality Control Using Machine Learning
## Purpose of the Project:
#### The project aims to build a machine learning pipeline to identify faulty steel plates by analyzing quality control data. This helps in automating the fault detection process in a production environment, improving efficiency and ensuring better product quality.

## Steps and Methodology:
### 1. Data Loading and Exploration:
#### - The dataset SteelPlatesFaults.csv is loaded using pandas.

#### - Basic exploratory data analysis (EDA) is performed:

#### -- Displaying the structure (df.info()), summary statistics (df.describe()), and checking for missing values.

#### -- A new column called Fault is created. If any of the last 7 columns (which represent different types of faults) have a value of 1, the product is marked as faulty.

#### -- Distribution of faulty vs. non-faulty products is visualized using a count plot.

### 2. Data Visualization:
#### - A heatmap is created to visualize correlations among the feature columns (excluding the last 8 columns).

#### - This helps to understand which features may be redundant or highly correlated.

### 3. Data Preprocessing:
#### - Features (X) and labels (y) are separated.

#### - Data is split into training and testing sets (80% training, 20% testing), using stratification to preserve class distribution.

#### - Features are standardized using StandardScaler to bring them onto a similar scale.

### 4. Model Training and Evaluation:
#### - A RandomForestClassifier is trained on the scaled training data.

#### - Predictions are made on the test data.

#### - A confusion matrix and classification report are printed to evaluate model performance (accuracy, precision, recall, F1-score).

### 5. Feature Importance Analysis:
#### - Feature importances from the trained Random Forest model are extracted.

#### - The top 10 most important features contributing to fault prediction are visualized using a horizontal bar plot.

### 6. Dimensionality Reduction and Visualization:
#### - Principal Component Analysis (PCA) is applied to reduce the dataset to 2 components for visualization.

#### - A scatter plot is created to visualize how products are distributed in the PCA-reduced space, colored by their fault status.

## Conclusion:
#### This project successfully demonstrates how machine learning, particularly Random Forests and PCA, can be applied to classify and analyze the quality of steel plates. The workflow includes preprocessing, modeling, evaluation, and visualization — making it a complete end-to-end quality control analysis pipeline.
