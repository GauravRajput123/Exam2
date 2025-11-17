import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Sample data (Replace with your actual dataset)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])  # XOR problem

# Create Neural Network
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  # First hidden layer with ReLU
model.add(Dense(1, activation='sigmoid'))            # Output layer with Sigmoid

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, verbose=0)

# Predict on training data
predictions = model.predict(X_train)
print(predictions)
