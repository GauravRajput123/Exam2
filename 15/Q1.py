import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\15\house_price_multiple.csv")
X = data[['feature1', 'feature2']]  # Replace with relevant features
y = data['price'] > data['price'].mean()  # Binary classification (Above/Below Average)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Build ANN model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=10)
