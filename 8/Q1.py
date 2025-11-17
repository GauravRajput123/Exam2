import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\8\news.csv")

# Features and labels
X = data['text']
y = data['category']

# Convert text to numerical vectors
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=0)

# Train Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("Predicted categories:")
print(predictions)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, predictions))
