import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\27\salary_position.csv")
X = data[['Level']]
y = data['Salary']

# Simple Linear Regression
model_linear = LinearRegression()
model_linear.fit(X, y)
linear_predictions = model_linear.predict(X)
linear_mse = mean_squared_error(y, linear_predictions)

# Polynomial Regression
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
poly_predictions = model_poly.predict(X_poly)
poly_mse = mean_squared_error(y, poly_predictions)

print(f"Linear MSE: {linear_mse}, Polynomial MSE: {poly_mse}")
