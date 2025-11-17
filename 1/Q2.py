import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the iris dataset from file
df = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\1\iris.csv")

# Encode species
labels = LabelEncoder().fit_transform(df['species'])

# Scatter plot
plt.scatter(df['sepal_length'], df['petal_length'], c=labels)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Scatter Plot of Iris Dataset')
plt.show()
