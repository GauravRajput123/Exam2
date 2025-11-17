import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\18\salary_position.csv")
X = data[['Level']]
y = data['Salary']

# Polynomial transformation
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the model
predictions = model.predict(X_test)
print(predictions)
