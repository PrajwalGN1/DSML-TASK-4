🏠 Housing Price Prediction

📌 Overview

This project builds a Linear Regression model to predict housing prices. It includes data preprocessing, feature scaling, encoding categorical variables, training/testing the model, evaluation, and visualization of results.

📂 Steps in the Project
1. Data Loading

Import dataset (Housing.csv) using pandas.

Display first rows and dataset info.

2. Data Cleaning

Check and remove duplicate rows.

Handle missing values:

Replace price with mean.

Replace no. of rooms with median.

Handle outliers in price using the 95th percentile cap.

3. Feature Engineering

One-hot encode categorical columns (e.g., Location, yes/no values).

Normalize numerical columns (area, no. of rooms) using MinMaxScaler.

4. Model Training

Split dataset into training (80%) and testing (20%).

Train Linear Regression model using scikit-learn.

Display model coefficients, intercept, and R² scores for training and testing sets.

5. Model Evaluation

Evaluate predictions using RMSE (Root Mean Square Error) and R² (Coefficient of Determination).

6. Visualization

Residual plot to check error distribution.

Scatter plot comparing Actual vs Predicted Prices, with a perfect prediction line.

📊 Results

Performance Metrics:

RMSE: Measures prediction error.

R²: Explains how much variance in prices is predicted by the model.

Insights:

A smaller RMSE means better accuracy.

Higher R² (close to 1) means strong predictive power.

⚙️ Requirements

Install required libraries before running the code:

pip install pandas numpy scikit-learn seaborn matplotlib

▶️ How to Run

Place the dataset Housing.csv in your working directory.

Run the Python script in Jupyter Notebook or IDE.

View results in the console and visualizations.

📌 Future Improvements

Try Regularization (Ridge/Lasso).

Use advanced models like Random Forest, XGBoost.

Hyperparameter tuning for better performance.

🌐 Availability & Contact

I am available for collaboration and research opportunities in Data Science & Machine Learning.
📩 Feel free to connect with me on 
www.linkedin.com/in/prajwal-g-n-3741a9332
