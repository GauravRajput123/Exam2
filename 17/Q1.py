import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\17\pima_indians_diabetes.csv")
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Bagging (Random Forest)
bagging_model = RandomForestClassifier(n_estimators=100)
bagging_model.fit(X_train, y_train)
bagging_predictions = bagging_model.predict(X_test)

# Boosting (Gradient Boosting)
boosting_model = GradientBoostingClassifier(n_estimators=100)
boosting_model.fit(X_train, y_train)
boosting_predictions = boosting_model.predict(X_test)

# Voting
voting_model = VotingClassifier(estimators=[
    ('rf', bagging_model), ('gb', boosting_model)], voting='hard')
voting_model.fit(X_train, y_train)
voting_predictions = voting_model.predict(X_test)

# Results
print("Bagging Accuracy:", accuracy_score(y_test, bagging_predictions))
print("Boosting Accuracy:", accuracy_score(y_test, boosting_predictions))
print("Voting Accuracy:", accuracy_score(y_test, voting_predictions))
