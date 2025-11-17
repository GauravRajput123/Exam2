import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\27\house_price_multiple.csv")

# Select features and target
X = data[['feature1', 'feature2', 'feature3']]
y = data['price']

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices
predictions = model.predict(X_test)

print("Predicted Prices:")
print(predictions)

print("\nActual Prices:")
print(y_test.values)
