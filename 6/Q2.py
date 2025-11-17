import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\6\employees.csv")

# Preprocessing – Remove missing values
data = data.dropna()

# Select feature (Income)
X = data[['income']]

# K-means clustering (4 groups)
kmeans = KMeans(n_clusters=4, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

# Print results
print("Employee Income Clusters:")
print(data)

# Plot clusters
plt.scatter(data['income'], data['Cluster'], c=data['Cluster'])
plt.xlabel('Income')
plt.ylabel('Cluster')
plt.title('Employee Income Clustering using K-Means')
plt.show()
