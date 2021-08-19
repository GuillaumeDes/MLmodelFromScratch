from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

iris_dataset = load_iris()
df = pd.DataFrame(data= np.c_[iris_dataset['data'], iris_dataset['target']],
                     columns= iris_dataset['feature_names'] + ['target'])
df.drop("target", axis=1, inplace=True)

# Scale data
X_scaled = StandardScaler().fit_transform(df)

# Compute Covariance
features = X_scaled.T
cov_matrix = np.cov(features)

# Eigendecomposition
values, vectors = np.linalg.eig(cov_matrix)

# explain variance along each principal component
explained_variances = [value / np.sum(values) for value in values]

# Project on two first principal component: this two new features contain 95% of the initial amount of variance
projected_1 = X_scaled.dot(vectors.T[0])
projected_2 = X_scaled.dot(vectors.T[1])

print(np.sum(explained_variances), "\n", explained_variances)
