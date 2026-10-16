import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform
from statistics import stdev
from sklearn.metrics import accuracy_score
import optuna

#1.Load data + split
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=====BaseLine=====")
# 2. Baseline pipeline: StandardScaler + LogisticRegression(C=1.0)
#    Fit, cross_val_score, print mean ± std
scaler = StandardScaler()
pipe_baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
cv_score = cross_val_score(pipe_baseline, X_train, y_train, cv=5)
print(f"mean and std: {cv_score.mean():.4f} ± {stdev(cv_score):.4f}")

print("\n=====GridSearchCV on LogisticRegression=====")
# 3. GridSearchCV on LogisticRegression
# Tune C over [0.001, 0.01, 0.1, 1 10, 100]
# Use a pipeline (standardscaler + LogisticRegression)
# cv=5
# print
# - best_params_
# - best_score_
# - the full grid results table (mean_test_score, param_C)
pipeline_logistic = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

param_grid = {
    'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    estimator=pipeline_logistic,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"best c value: {grid_search.best_params_['logreg__C']}")
print(f"best accuracy score: {grid_search.best_score_:.4f}")
print(f"grid results: {grid_search.cv_results_}")

# 4. Apply the best model to the test set
#    Print test accuracy
#    Compare to baseline
y_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")


print("\n=====RandomizedSearchCV on RandomForest=====")
# 5. RandomizedSearchCV on RandomForest
#    Tune:
#      n_estimators: randint(50, 300)
#      max_depth: randint(3, 20)
#      max_features: uniform(0.1, 0.9)   # fraction of features
#    n_iter=20, cv=5
#    Print best_params_ and best_score_
pipeline_rd_search = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

param_dist = {
    'model__n_estimators': randint(50, 300),
    'model__max_depth': randint(3, 20),
    'model__max_features': uniform(0.1, 0.9)
}

rd_search = RandomizedSearchCV(
    estimator=pipeline_rd_search,
    param_distributions=param_dist,
    n_iter=20, 
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
rd_search.fit(X_train, y_train)
print(f"best parameters: {rd_search.best_params_}")
print(f"best score: {rd_search.best_score_}")

# 6. Apply the best forest to the test set
#    Print test accuracy
#    Compare to a default RandomForestClassifier (n_estimators=100)
y_pred = rd_search.predict(X_test)
test_accuracy_rd_search = accuracy_score(y_test, y_pred)

print("=====RandomForestClassifier vs RandomizedSearch====")
print(f"RandomizedSearchCV test accuracy: {test_accuracy_rd_search:.4f}")
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])
pipeline_rf.fit(X_train, y_train)
test_accuracy_rf = accuracy_score(y_test, pipeline_rf.predict(X_test))
print(f"RandomForestClassifier test accuracy: {test_accuracy_rf:.4f}")




print("\n=====Optuna====")
# 7. (BONUS) — try Optuna
#    pip install optuna first
#    import optuna
#    Define objective: tune Ridge's alpha and LogisticRegression's C
#    n_trials=30
#    Print study.best_params and study.best_value
# Study 1 — Ridge
def objective_ridge(trial):
    alpha = trial.suggest_float('alpha', 0.001, 100, log=True)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RidgeClassifier(alpha=alpha))
    ])
    return cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()

study_ridge = optuna.create_study(direction='maximize')
study_ridge.optimize(objective_ridge, n_trials=30)
print("Ridge best alpha:", study_ridge.best_params)
print("Ridge best score:", study_ridge.best_value)

# Study 2 — LogisticRegression
def objective_lr(trial):
   c = trial.suggest_float('C', 0.001, 20, log=True)
   pipe = Pipeline([('scaler', StandardScaler()),
                     ('model', LogisticRegression(C=c, max_iter=10000))])
   return cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()

study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objective_lr, n_trials=30)
print("LR best C:", study_lr.best_params)
print("LR best score:", study_lr.best_value)