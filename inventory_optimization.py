# inventory_optimization.py

"""
Inventory Optimization Using Machine Learning

This script utilizes an XGBoost model for demand forecasting to optimize inventory management. 
It includes data preprocessing, model training, and visualization of the predicted vs. actual demand.

Assumptions: 
- The dataset has features relevant to inventory management, including past demand and other predictors.
- A mock dataset will be generated for demonstration purposes.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate a mock dataset
np.random.seed(42)
data_size = 1000
data = pd.DataFrame({
    'feature1': np.random.rand(data_size) * 100,  # Example feature
    'feature2': np.random.rand(data_size) * 50,   # Another feature
    'demand': np.random.randint(1, 100, size=data_size)  # Demand as target variable
})

# Data Preprocessing
data.fillna(method='ffill', inplace=True)  # Forward fill any missing values

# Feature selection
features = data[['feature1', 'feature2']]
target = data['demand']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the XGBoost model
model = XGBRegressor(objective='reg:squarederror')

# Fit the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse:.2f}')

# Visualize the predictions vs actual demand
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title('Actual vs Predicted Demand')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # Diagonal line
plt.grid()
plt.show()
