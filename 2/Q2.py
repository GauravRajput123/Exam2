import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load dataset correctly
df = pd.read_csv(r"C:\Users\gaura\Downloads\ML\ML\2\wholesale_customers.csv")

X = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clustering = AgglomerativeClustering(n_clusters=3)
labels = clustering.fit_predict(X_scaled)

df['Cluster'] = labels

print(df.head())
