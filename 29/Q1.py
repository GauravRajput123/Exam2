import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Iris Dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\29\iris.csv")   # Make sure iris.csv is in same folder

# Encode labels
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

X = df.drop("species", axis=1)   # 4 features: SL, SW, PL, PW
y = df["species"]

# -----------------------------
# PCA: Reduce 4D → 2D
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42
)

# -----------------------------
# Train SVM Model
# -----------------------------
model = SVC(kernel="rbf")
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))

# -----------------------------
# Predict new flower
# -----------------------------
new_flower = [[5.4, 3.2, 4.5, 1.5]]  # Example input

# Convert new flower from 4D → 2D using same PCA
new_flower_pca = pca.transform(new_flower)

prediction = model.predict(new_flower_pca)
print("Predicted Flower:", le.inverse_transform(prediction)[0])
