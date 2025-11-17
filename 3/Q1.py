import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\3\house_price.csv")

# Select features (independent variables)
X = data[['size_sqft', 'bedrooms', 'age_years']]   # Actual feature names

# Target variable
y = data['price']

# Split dataset into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict using test data
predictions = model.predict(X_test)

print("Predicted Prices:")
print(predictions)

# Model accuracy (optional)
print("\nR-squared Score:", model.score(X_test, y_test))
