import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\11\banknote_authentication.csv")

# Features and target
X = data.drop(columns=['class'])  # All columns except 'class'
y = data['class']                 # Target variable: 0 = genuine, 1 = forged

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Train Decision Tree classifier
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
