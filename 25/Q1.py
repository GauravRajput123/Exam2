import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\25\house_price.csv")
X = data[['area']]  # Independent variable
y = data['price']       # Target variable

# Polynomial transformation (degree 3)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.3, random_state=0
)

# Train Linear Regression model on polynomial features
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)
print("Predicted Prices:", predictions)
