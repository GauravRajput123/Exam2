import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\22\house_prices.csv")

# Select single feature for Simple Linear Regression
X = data[['feature1']]
y = data['price']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Show results
print("Predicted Prices:")
print(predictions)
