import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from statistics import mean, stdev
from tabulate import tabulate
import pandas as pd

#1.Load +split (same as before)
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2.Train a sinlge decisionTree (for comparision)
#use random_state=42
model_d = DecisionTreeClassifier(random_state=42)
model_d.fit(X_train, y_train)
y_d_train_pred = model_d.predict(X_train)
y_d_test_pred = model_d.predict(X_test)


#3.Train a RandomForestClassifier with n_estimators=100
#use random_state=42
model_r = RandomForestClassifier(n_estimators=100, random_state=42)
model_r.fit(X_train,y_train)
y_r_train_pred = model_r.predict(X_train)
y_r_test_pred = model_r.predict(X_test)

#4.Compute train and test accuracy for both models

acc_d_train = accuracy_score(y_train, y_d_train_pred)
acc_d_test = accuracy_score(y_test, y_d_test_pred)
acc_r_train = accuracy_score(y_train, y_r_train_pred)
acc_r_test = accuracy_score(y_test, y_r_test_pred)

headers = ["Model", "Train acc", "Test Acc"]
content = [
    ["Decision Tree", acc_d_train, acc_d_test ],
    ["Random Forest", acc_r_train, acc_r_test]
]

print(tabulate(content, headers=headers))

#5.Now vary_estimators: [1, 5, 10, 50, 100, 500]
#For each train a forest, compute test accuracy
#print a table
#what happens to test accuracy as n_estimators grows
#ans:accuracy increased and been constant after a specific n_estimator value
content = []
for n in [1, 5, 10, 50, 500]:
    model = RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    content.append([n, accuracy])

content.insert(4, [100, acc_r_test])

headers = ["n_esti", "accuracy"]

print("\nAccuracy of random forest as n_estimators increases: ")
print(tabulate(content, headers=headers))

#6. Use 5-fold cross-validation to compare
acc_crossval_d = cross_val_score(model_d, X, y, cv=5, scoring='accuracy')
acc_crossval_r = cross_val_score(model_r, X, y, cv=5, scoring='accuracy')

mean_acc_cv_d = mean(acc_crossval_d)
std_acc_cv_d = stdev(acc_crossval_d)

mean_acc_cv_r = mean(acc_crossval_r)
std_acc_cv_r = stdev(acc_crossval_r)

content = [["Decision Tree", mean_acc_cv_d, std_acc_cv_d],
           ["Random Forest", mean_acc_cv_r, std_acc_cv_r]]
headers = ["model", "mean", "std"]

print("\nmean and std using 5-fold crossvalidation: ")
print(tabulate(content, headers=headers))


#7.Print the top100 most important features from the random forest
#use model.feature_importances_ and data.feature_names
#Sort and print top 10
important_feat = model.feature_importances_
cols = data.feature_names

feature_imp = pd.DataFrame({'Feature': data.feature_names,
                            'Importance':model_r.feature_importances_})

top_10_features = feature_imp.sort_values(by='Importance', ascending=False).head(10)

print("\nTop 10 most important features:")
print(top_10_features.to_string(index=False))

#8. USE OOB score
# Train a random forest with oob_score=True
# Print model.oob_score_
model_oob = RandomForestClassifier(oob_score=True)
model_oob.fit(X_train, y_train)
print("Model oob score:", model_oob.oob_score_)