import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\9\universalbank.csv")

# Features and target
X = data.drop(columns=['Personal Loan'])  # All features except target
y = data['Personal Loan']  # Target column (1: approved, 0: not approved)

# Optional: Feature scaling (recommended for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=0
)

# Train Linear SVM
model = SVC(kernel='linear', random_state=0)
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
