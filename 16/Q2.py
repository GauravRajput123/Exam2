import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset from CSV (50 rows)
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\16\boston_housing.csv")

# Use 'RM' as feature and 'MEDV' as target
X = data[['RM']]  # Average number of rooms
y = data['MEDV']  # Median value of homes

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print predictions
print("Predictions:", predictions)

# Visualize results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, predictions, color='red', linewidth=2, label='Predicted')
plt.xlabel('Average number of rooms (RM)')
plt.ylabel('Median value (MEDV)')
plt.title('Simple Linear Regression - Boston Housing')
plt.legend()
plt.show()
