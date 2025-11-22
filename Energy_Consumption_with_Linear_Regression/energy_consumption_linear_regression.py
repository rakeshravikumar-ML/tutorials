# Predicting Energy Consumption using Linear Regression (OLS)
# ----------------------------------------------------------
# Beginner-friendly example using scikit-learn
# Synthetic dataset for demonstration

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------------------
# 1. Create a small synthetic dataset
# -------------------------------------------
np.random.seed(42)

# Let's say energy depends on temperature + humidity + house_size
temperature = np.random.uniform(5, 35, 200)            # in Celsius
humidity = np.random.uniform(20, 90, 200)              # percentage
house_size = np.random.uniform(600, 3500, 200)         # sq ft

# True relationship (unknown to model)
energy = 3.5 * temperature + 1.2 * humidity + 0.05 * house_size + np.random.normal(0, 10, 200)

df = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "house_size": house_size,
    "energy_consumption": energy
})

# -------------------------------------------
# 2. Split into train/test sets
# -------------------------------------------
X = df[["temperature", "humidity", "house_size"]]
y = df["energy_consumption"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------
# 3. Train the Linear Regression model (OLS)
# -------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------------------
# 4. Evaluate the model
# -------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# -------------------------------------------
# 5. Predict for a new house
# -------------------------------------------
sample = pd.DataFrame({
    "temperature": [22],
    "humidity": [55],
    "house_size": [1500]
})

prediction = model.predict(sample)[0]
print(f"\nPredicted Energy Consumption for sample house: {prediction:.2f}")
