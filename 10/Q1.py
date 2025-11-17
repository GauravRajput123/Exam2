import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\10\iris.csv")
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Map species to numeric colors
color_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}

# Plot the PCA result
plt.figure(figsize=(8,6))
plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=data['species'].map(color_map),
    cmap='viridis',
    edgecolor='k',
    s=100
)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')
plt.colorbar(ticks=[0,1,2], label='Species')
plt.show()
