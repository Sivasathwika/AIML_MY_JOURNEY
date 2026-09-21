# Lesson6 : Your first Real ML Pipeline

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#1.Load the data
data = load_diabetes()
X = data.data       #features
y = data.target     #target

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Feature names:", data.feature_names)

#2.Split into train and test sets
#test_size = 0.2, random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3.Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

#4.Predict on both training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#5. Compute MSE on both
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

#6.Print the results
# trainign MSE
# test MSE
# is the test MSE higher or lower than training MSE?
print(f"training MSE: {mse_train:.4f}")
print(f"test MSE: {mse_test:.4f}")


#7.Print model.coef_ and model.intercept_
print("\n Model coefficients (one per feature):", model.coef_)
print("Model intercept (bias):", model.intercept_)
