import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\8\tennis.csv")

# Encode categorical columns
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])

# Features and target
X = data.drop(columns=['play'])
y = data['play']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions on test set
pred_test = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred_test))

# Predict for a new sample (must be DataFrame with same columns)
new_sample = pd.DataFrame([[2, 1, 1, 0]], columns=X.columns)
print("Prediction for new sample:", model.predict(new_sample))
