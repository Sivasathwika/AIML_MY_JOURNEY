import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabulate import tabulate

#1.Load breast cancer (30 features, binary target)
data = load_breast_cancer()
X, y = data.data, data.target

print("Original shape:", X.shape)

#2.Scale the data (CRITICAL FOR PCA)
#standardscaler, fit_transform on X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#3. Fit PCA with no componentt limit
# pca = PCA()
# pca.fit(X_scaled)
# print explained_varaince_ratio (first 10 values)
# Print cummulative sum of explained variance
# How many components to reach 95%?
pca = PCA()
pca.fit(X_scaled)
ex_var_ratio = pca.explained_variance_ratio_
print(f"explained variance ratio: {ex_var_ratio[:10]}")
print(f"Cummulative  :{pca.explained_variance_ratio_.sum()}")

#4.Fit PCA with n_components=2
# pca2 = PCA(n_components=2)
# X_pca = pca2.fit_transform(X_scaled)
#print the shape of X_pca
# what percentage of variance do the 2 components capture?
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)
print(X_pca.shape)
print(f"for 2 components variance captured: {pca2.explained_variance_ratio_.sum()}")

#5.Visualize:
# Plot X_pca as a scatter (c=y, cmap='coolwarm')
# Are malignant and benign seperable in 2D? ans: yes
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm')
plt.show()

#6.Test if PCA hurts model performance:
#   -split data: X_pca and y into train/test
#   -Train logistic regression on X_pca
#   -compute accuracy
#   -compare to logistic regression on the full scaled X
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model_lr_pca = LogisticRegression()
model_lr_pca.fit(X_pca_train, y_train)
y_pca_pred = model_lr_pca.predict(X_pca_test)
acc_pca = accuracy_score(y_test, y_pca_pred)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model_lr_scaled = LogisticRegression()
model_lr_scaled.fit(X_train, y_train)
y_scaled_pred = model_lr_scaled.predict(X_test)
acc_scaled = accuracy_score(y_test, y_scaled_pred)

print(f"accuracy of model with pca: {acc_pca:0.4f}")
print(f"accuracy of model without pca(only scaling): {acc_scaled:0.4f}")

#7.Sweep n_components: [2, 5, 10, 15, 20, 25, 30]
# For each, fit PCA, train logistic regression, record test accuracy
# print a table
content = []

for n in [2, 5, 10, 15, 20, 25, 30]:
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    ex_var_ratio = pca.explained_variance_ratio_.sum()
    content.append([n, acc, ex_var_ratio])

headers = ["n_comp", "test accuracy", "variance captured"]

print(tabulate(content, headers=headers))

# ------------------------------------------------------------------
# 8. Cumulative explained variance plot
# ------------------------------------------------------------------
# Cumulative sum of the ratios = "how much total variance is
# retained if I keep the first K components?"
cumulative = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative) + 1), cumulative,
         marker='o', color='steelblue', label='Cumulative variance')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
plt.axvline(x=10, color='green', linestyle=':', label='k=10')

plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA — how many components do we need?')
plt.xticks(range(1, len(cumulative) + 1, 2))
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Find exactly how many components reach 95%
n_95 = np.argmax(cumulative >= 0.95) + 1
print(f"Components needed to reach 95% variance: {n_95}")