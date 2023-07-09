# courtesy of Bernard
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# Constants
n_samples = 4000
n_features = 4
n_clusters = 4

# Generate data (SQIs)
X, y = make_blobs(n_samples=n_samples,
                  n_features=n_features,
                  centers=n_clusters,
                  cluster_std=0.60,
                  random_state=0)

# Train PCA
pca = PCA(n_components=2)
pca.fit(X)

# Create dataframe for visualisation
# Add SQIs
data = pd.DataFrame(data=X,
                    columns=['sqi_%s' % i for i in range(n_features)])
# Add PCA projections
data[['pca_x', 'pca_y']] = pca.transform(X)
# Add cluster id
data['cluster'] = y

# Compute stats
table = data.groupby('cluster').agg(['mean', 'std'])

# Show
print("\ndata:")
print(data)
print("\ntable:")
print(table)

# Plot
for k in data.cluster.unique():
    plt.scatter(data.loc[data.cluster == k, 'pca_x'],
                data.loc[data.cluster == k, 'pca_y'],
                marker='.', s=10, alpha=0.7)
plt.grid(True)
plt.show()
