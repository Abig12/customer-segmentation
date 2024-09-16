
import pandas as pd  # We’ll use pandas for data handling
import matplotlib.pyplot as plt  # matplotlib and seaborn for visualization
import seaborn as sns  # scikit-learn for machine learning.
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('Mall_Customers.csv')  # useing pandas to load the CSV file.

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, init='k-means++',
                max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_scaled)

# Assign the cluster labels to the original dataset
df['Cluster'] = kmeans.labels_

score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {score}')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
           c=df['Cluster'], cmap='Set1', s=50)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending Score')
plt.show()
