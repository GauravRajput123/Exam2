import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('salary_positions.csv')
X = data[['level']]  # Feature: employee level
y = data['salary']   # Target: salary

# ------------------------
# Simple Linear Regression
# ------------------------
model_linear = LinearRegression()
model_linear.fit(X, y)

# Predictions on training data
linear_predictions = model_linear.predict(X)
linear_mse = mean_squared_error(y, linear_predictions)

# Predict salaries for level 11 and 12
linear_salary_pred = model_linear.predict([[11], [12]])

# ------------------------
# Polynomial Regression (degree 3)
# ------------------------
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model_poly = LinearRegression()
model_poly.fit(X_poly, y)

# Predictions on training data
poly_predictions = model_poly.predict(X_poly)
poly_mse = mean_squared_error(y, poly_predictions)

# Predict salaries for level 11 and 12
X_new_poly = poly.transform([[11], [12]])
poly_salary_pred = model_poly.predict(X_new_poly)

# ------------------------
# Results
# ------------------------
print(f"Linear Regression MSE: {linear_mse:.2f}")
print(f"Polynomial Regression MSE: {poly_mse:.2f}\n")

print(f"Linear Regression Predicted Salaries: Level 11 = ${linear_salary_pred[0]:,.2f}, Level 12 = ${linear_salary_pred[1]:,.2f}")
print(f"Polynomial Regression Predicted Salaries: Level 11 = ${poly_salary_pred[0]:,.2f}, Level 12 = ${poly_salary_pred[1]:,.2f}")
