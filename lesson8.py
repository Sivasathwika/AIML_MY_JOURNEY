import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#1.Load data
data = load_breast_cancer()
X = data.data
y = data.target

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", data.target_names)   #['malignant', 'benign']
print("class counts:", np.bincount(y)) #how many of each class

#2.split (test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3.Create LogisticRegression model and fit it
#NOTE: unlike Linear regresion, logistic regression sometimes needs 
#      a higher max_iter (try max_iter=10000 if you get a warning)
model = LogisticRegression(max_iter=10000)
model.fit(X_train,y_train)

#4.Predict on the test set
y_pred = model.predict(X_test)

#5. Compute accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


#6. Print the model's learned coefficients and intercept
print("\n Model coefficients (one per feature):", model.coef_)
print("Model intercept (bias):", model.intercept_)

#7.IMPORTANT: predict PROBABILITIES, not just classes
#model.predict_proba(X_test) returns a 2D array
#column 0 = probability of class 0
#column 1 = probability of class 1
proba = model.predict_proba(X_test)
print("Probabilities shape:",proba.shape)
print(proba)

#8. Print the first 5 predictions along with their probablities
# i.using range
for i in range(5):
    pred = y_pred[i]
    prob = proba[i, 1]   #probability of class1
    print(f"Example {i}: prediction = {pred}, P(class1)= {prob:.4f}")

"""
# ii.using zip
for i, (pred, prob) in enumerate(zip(y_pred[:5], proba[:5])):
    print(f"Example {i}: prediction={pred}, P(class 1)={prob[1]:.4f}")
"""