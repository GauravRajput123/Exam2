import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\7\weather.csv")

# Convert categorical features to numeric
data_encoded = data.apply(lambda col: col.astype('category').cat.codes)

# Features and target
X = data_encoded.drop(columns=['play'])
y = data_encoded['play']

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
