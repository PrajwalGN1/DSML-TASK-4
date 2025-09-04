import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#loading the cs file
df=pd.read_csv(r"C:\Users\DELL\Downloads\Housing (2).csv")

#print the top 5 rows
print("Top 5 rows are : \n",df.head())

#Print the info of dataset
print("The dataset info : \n")
df.info()

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Remove duplicates
#df.drop_duplicates(inplace=True)

# Handle missing values
print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")

# Fill missing numerical values with column mean
df['price'].fillna(df['price'].mean, inplace=True)
df['no. of rooms'].fillna(df['no. of rooms'].median,inplace =True)

print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")

#Handling outliers
upper_limit = df['price'].quantile(0.95)
df['price'] = np.where(df['price'] > upper_limit ,upper_limit, df['price'])
df['price']


# One-hot encode categorical columns (like Location, yes/no columns)
df = pd.get_dummies(df, drop_first=True)
df.head()

from sklearn.preprocessing import MinMaxScaler

# Normalise the numerical columns
scaler = MinMaxScaler()
df[['area','no. of rooms']]=scaler.fit_transform(df[['area','no. of rooms']])

# Define features and target variables
X = df.drop('price',axis=1)
y = df['price']

# Split the datamodel into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state =42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Initialize the training model
model= LinearRegression()
model.fit(X_train,y_train)

# Display results
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)
print("Training R²:", model.score(X_train, y_train))
print("Testing R²:", model.score(X_test, y_test))

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

print(f"RMSE :{rmse: .2f}")
print(f"r^2 : {r2 :.2f}")

# Visualization

# Residual plot
residuals=y_test-y_pred
plt.figure(figsize=(7,4))
sns.histplot(residuals, kde=True ,bins=30 , color='blue')
plt.title("Distribution of residuals")
plt.xlabel("Residuals")
plt.show()

# Scatter plot of actual vs predictions

plt.figure(figsize=(7,4))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue", label="Predictions")

# Add perfect prediction line (y = x)
max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Prediction")

plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.show()

