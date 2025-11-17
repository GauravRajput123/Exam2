import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\4\mall_customers.csv")

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

# Print clustered data
print(data[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']])

# Visualization
plt.scatter(X['Annual Income (k$)'], 
            X['Spending Score (1-100)'], 
            c=data['Cluster'])

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-means Clustering of Mall Customers')
plt.show()
