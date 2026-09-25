import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#1.Load data
data = load_diabetes()
X, y = data.data, data.target

#2.Same split as lesson 6 - reproduce the numbers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2
                                                    , random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_test_pred = model.predict(X_test)

#3.Compute both metrics on the test set
# MSE and R²
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print("single split r2:", r2)

#4.Try three different random_state values
#For each split train and record test R² 
#random_state = 0, 1, 2
#Print each R²
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model0 = LinearRegression()
model0.fit(X_train, y_train)
y_test_pred = model0.predict(X_test)
r2_0 = r2_score(y_test, y_test_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model1 = LinearRegression()
model1.fit(X_train, y_train)
y_test_pred = model1.predict(X_test)
r2_1 = r2_score(y_test, y_test_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
model2 = LinearRegression()
model2.fit(X_train, y_train)
y_test_pred = model2.predict(X_test)
r2_2 = r2_score(y_test, y_test_pred)

print("R2 value at random state 0 :", r2_0)
print("R2 value at random state 1 :", r2_1)
print("R2 value at random state 2 :", r2_2)


#5.Now do 5-fold cross-validation
# cross_val_score(model, X, y, cv=5, scoring='r2')
# scores is and array of numbers - one per fold, here cv=5 so 5 numbers array 
#Print: each fold score, the mean, and the std

scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(scores)
print("cross_val_scores R2 mean:",scores.mean())
print(scores.std())

#6.Compare 
# Is the mean cross-val R² close to the single-split R²
#ans: 
# how much do the individual fold scores vary?

