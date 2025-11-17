import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\18\diabetes.csv")
X = data[['Glucose', 'BloodPressure']]  # Choose relevant features

# K-means clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(X['Glucose'], X['BloodPressure'], c=data['Cluster'])
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
plt.title('K-means Clustering on Diabetes Dataset')
plt.show()
