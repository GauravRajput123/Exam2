import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\10\iris.csv")

# Convert categorical species into numeric values
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])

# Create Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(
    data['sepal_length'],
    data['petal_length'],
    c=data['species'],  # numeric species as color
    cmap='viridis',
    edgecolor='k',
    s=100
)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Scatter Plot of Iris Dataset with Numeric Species')
plt.colorbar(ticks=[0,1,2], label='Species')  # Shows numeric species mapping
plt.show()
