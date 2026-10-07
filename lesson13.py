import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

#1.Load data (diabetes - same as lesson 6)
data = load_diabetes()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the data (linear models like scaled input)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

#2.Baseline: plain linear regression
#fit, predict, print test MSE and R²
#print the largest absolute coefficient
model_lr = LinearRegression()
model_lr.fit(X_train_s, y_train)
y_pred = model_lr.predict(X_test_s)

mse_lr = mean_squared_error(y_test, y_pred)
r_square_lr = r2_score(y_test, y_pred)
print(f"Baseline Linear Regression - Test MSE: {mse_lr:0.4f}, R²: {r_square_lr:0.4f}")

print(f"Largest absolute coefficient: {np.max(np.abs(model_lr.coef_))}")
print()
#3. Ridge with alpha=1.0
#Fit, predict, print test MSE and R²
#print the largest absolute coefficient
#print how many coefficients became exactly zero
model_rd = Ridge(alpha=1.0)
model_rd.fit(X_train_s, y_train)
y_pred_rd = model_rd.predict(X_test_s)
mse_rd = mean_squared_error(y_test, y_pred_rd)
r_square_rd = r2_score(y_test, y_pred_rd)
print(f"Ridge Regression - Test mse: {mse_rd:0.6f}, R²: {r_square_rd:0.4f}")
print(f"Largest absolute coefficient: {np.max(np.abs(model_rd.coef_))}")
print(f"coefficients that are zero: {np.sum(model_rd.coef_ == 0)}")
print()
#4.Lasso with alpha=0.1
#Fit, predict, print test MSE and R²
#print the largest absolute coefficient
#print how many coefficients became exactly zero
model_ls = Lasso(alpha=0.1)
model_ls.fit(X_train_s, y_train)
y_pred_ls = model_ls.predict(X_test_s)
mse_ls = mean_squared_error(y_test, y_pred_ls)
r_square_ls = r2_score(y_test, y_pred_ls)
print(f"Lasso Regression - Test mse: {mse_ls:0.6f}, R²: {r_square_ls:0.4f}")
print(f"Largest absolute coefficient: {np.max(np.abs(model_ls.coef_))}")
print(f"coefficients that are zero: {np.sum(model_ls.coef_ == 0)}")

print()
#5. Compare coefficient magnitudes
#For linearregression, ridge, lasso, print the sum of |coef|.
#which is smallest? which is largest?

print(f"Sum of |coef| for Linear Regression: {np.sum(np.abs(model_lr.coef_)):0.4f}")
print(f"Sum of |coef| for Ridge Regression: {np.sum(np.abs(model_rd.coef_)):0.4f}")
print(f"Sum of |coef| for Lasso Regression: {np.sum(np.abs(model_ls.coef_)):0.4f}")

sums = {
    'Linear':  np.sum(np.abs(model_lr.coef_)),
    'Ridge':   np.sum(np.abs(model_rd.coef_)),
    'Lasso':   np.sum(np.abs(model_ls.coef_)),
}
print(f"Smallest sum: {min(sums, key=sums.get)}")
print(f"Largest sum:  {max(sums, key=sums.get)}")
print()
#6.Sweep alpha for ridge: [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
#For each , print test R² and sum of |coef|
# what happens as alpha grows?
for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = Ridge(alpha=alpha)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    r_square = r2_score(y_test, y_pred)
    print(f"Ridge Regression with alpha={alpha} - Test R²: {r_square:0.4f}, Sum of |coef|: {np.sum(np.abs(model.coef_)):0.4f}")
print()
#7. Same sweep for Lasso: [0.001, 0.01, 0.1, 1, 2, 5, 10, 50, 100]
#For each , print test R² and sum of |coef|
# what happens as alpha grows?
for alpha in [0.001, 0.01, 0.1, 1, 2, 5, 10, 50, 100]:
    model = Lasso(alpha=alpha)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    r_square = r2_score(y_test, y_pred)
    print(f"Lasso Regression with alpha={alpha} - Test R²: {r_square:0.4f}, Sum of |coef|: {np.sum(np.abs(model.coef_)):0.4f}, zero coefficients: {np.sum(model.coef_ == 0)}")
print()
#8. Use RidgeCV to find the best alpha automatically
#print the chosen alpha and test R²
model_ridgecv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], store_cv_results=True)
model_ridgecv.fit(X_train_s, y_train)
y_pred_ridgecv = model_ridgecv.predict(X_test_s)
print(f"Best alpha for Ridge Regression: {model_ridgecv.alpha_}")
print(f"Test R² for Ridge Regression: {r2_score(y_test, y_pred_ridgecv):0.4f}")

