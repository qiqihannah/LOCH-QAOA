from sklearn.cluster import KMeans
import numpy as np

# Standardize the data to have mean=0 and variance=1
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

file_name = "paintcontrol"
# Sample data: 80 samples with 2 features
df = pd.DataFrame()
df = pd.read_csv("data/" + file_name + ".csv", dtype={"time": float, "rate": float})
ori_data = np.array(df.values.tolist())
print(ori_data)
indices = ori_data[:,0]
list_data = ori_data[:, 1:3]

kmeans = KMeans(n_clusters=46)
kmeans.fit(list_data)

# Get the cluster centers
centroids = kmeans.cluster_centers_

# Calculate distances from each instance to the cluster centers
distances = np.linalg.norm(list_data[:, np.newaxis] - centroids, axis=2)

# Find the indices of the closest instances
selected_indices = np.argmin(distances, axis=0)

# Get the original indices of the selected instances
selected_instance_indices = indices[selected_indices].astype(int)

selected_cases = ori_data[selected_indices, :]

# Print the indices of the selected instances
print("Indices of selected instances:", selected_instance_indices.tolist())

sorted_selected_cases = selected_cases[np.argsort(selected_cases[:, 0])]
print(sorted_selected_cases)

selected_df = pd.DataFrame(sorted_selected_cases[:, 1:], columns=['time', 'rate'])
selected_df.to_csv("data/sampled_paintcontrol.csv")