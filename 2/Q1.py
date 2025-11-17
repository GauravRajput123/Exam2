import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and clean data
df = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\2\house_prices.csv")

df_cleaned = df.dropna()

# Prepare data
X = df_cleaned[['size_sqft', 'bedrooms', 'age_years']]
y = df_cleaned['price']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print(f"R-squared: {model.score(X_test, y_test):.4f}")

# Predict new house
new_house = pd.DataFrame([[1800, 3, 4]], columns=['size_sqft', 'bedrooms', 'age_years'])
predicted_price = model.predict(new_house)[0]
print(f"Predicted price: ${predicted_price:,.2f}")
