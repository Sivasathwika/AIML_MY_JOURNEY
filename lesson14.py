import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd

#1. Load iris dataset (has 3 species, but we ignore labels)
data = load_iris()
X = data.data
y_true = data.target

print("X shape:", X.shape)
print("True classes:", np.unique(y_true))

#2.scale features (K-Means is distance --based, so we need to scale features)
#Standardscaler, fit_tramsforms on X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#3. Fit KMeans with n_clusters=3 (we know there are 3 species)
# use random_state=42, n_init=10
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X_scaled)

#4.Print cluster labels (first 10), centroids, inertial
print("Cluster labels (first 10):", model.labels_[:10])
print("Centroids:\n", model.cluster_centers_)
print("Inertia:", model.inertia_)

#5.Crosstab between kmeans.labels_ and y_true_
#Did K-Means find groups matching the true species?
df = pd.DataFrame({'KMeans': model.labels_, 'True': y_true})
crosstab = pd.crosstab(df['KMeans'], df['True'])
print("Crosstab:")
print(crosstab)

#6.Elbow method : for k in [1 .. 8], fit, record inertia
#print each k and inertia 
# where is the elbow?
#7.Silhouette score for k=2....8
#print score for each k
#which k is best?
for k in range(1,9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    print(f"for k={k} ")
    print(f"inertia is : {model.inertia_}")
    if(k>=2):
        sil_score = silhouette_score(X_scaled, model.labels_)
        print(f"silhouette score is : {sil_score}")

print("\n\n\n")
#8. Fit KMeans on UNSCALED X, compare inertia
# why is unscaled worse?
for k in range(1,9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    print(f"for k={k} ")
    print(f"inertia is : {model.inertia_}")
   
    