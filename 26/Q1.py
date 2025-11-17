import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\26\diabetes_indian.csv")

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Find optimal K
accuracies = []
k_values = range(1, 21)

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, pred))

optimal_k = k_values[accuracies.index(max(accuracies))]
print("Optimal K =", optimal_k)

# Train final model
final_model = KNeighborsClassifier(n_neighbors=optimal_k)
final_model.fit(X_train, y_train)

# Predict a new patient
new_patient = [[3, 130, 75, 28, 110, 33.5, 0.56, 30]]
prediction = final_model.predict(new_patient)

print("New patient diabetic? ->", prediction[0])
