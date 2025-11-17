import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\28\iris.csv")

# Convert species to numbers
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

X = df.drop("species", axis=1)
y = df["species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

kernels = ["linear", "poly", "rbf", "sigmoid"]
accuracies = {}

# Test all kernels
for k in kernels:
    model = SVC(kernel=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracies[k] = accuracy_score(y_test, pred)

print("Accuracy for each SVM kernel:")
for k, acc in accuracies.items():
    print(k, ":", round(acc, 4))

# Train final model using best kernel
best_kernel = max(accuracies, key=accuracies.get)
print("\nBest Kernel =", best_kernel)

final_model = SVC(kernel=best_kernel)
final_model.fit(X_train, y_train)

# Predict a new flower
new_data = [[5.4, 3.2, 4.5, 1.5]]
prediction = final_model.predict(new_data)

print("\nPredicted Flower Type:", le.inverse_transform(prediction)[0])
