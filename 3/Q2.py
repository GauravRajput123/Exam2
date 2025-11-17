import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\3\crash.csv")

# Features and label
X = data[['age', 'speed']]
y = data['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Build model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
print("Predicted Survival:", predictions)
