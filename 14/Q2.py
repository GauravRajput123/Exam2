import pandas as pd

# 1. Create a sample dataset with some null values
data = pd.DataFrame({
    'Feature1': [1, 2, None, 4, 5],
    'Feature2': [None, 'B', 'C', 'D', 'E'],
    'Feature3': [10, None, 30, 40, None]
})

# 2. Display the dataset
print("Original Dataset:")
print(data)

# 3. Check for null values
print("\nNull values in each column:")
print(data.isnull().sum())

# 4. Remove rows containing any null values
data_cleaned = data.dropna()
print("\nDataset after removing null values:")
print(data_cleaned)
