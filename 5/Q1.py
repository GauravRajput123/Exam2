import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\5\fuel_consumption.csv")

# Multiple features
X = data[['feature1', 'feature2', 'feature3']]
y = data['consumption']

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("Predicted Fuel Consumption:")
print(predictions)
