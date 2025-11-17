import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\9\boston_houses.csv")

# Features and target
X = data[['RM']]  # Number of rooms
y = data['Price']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# Lasso Regression
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)

# Predict price for a house with 5 rooms
sample = pd.DataFrame([[5]], columns=['RM'])
ridge_pred = ridge_model.predict(sample)
lasso_pred = lasso_model.predict(sample)

print("Ridge Prediction for 5 rooms: $", ridge_pred[0])
print("Lasso Prediction for 5 rooms: $", lasso_pred[0])
