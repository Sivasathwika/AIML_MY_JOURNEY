import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

# 1. Load + split (same as before)
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=====Baseline-LogisticRegression=====")
# 2. Baseline — LogisticRegression (best so far)
# your code — fit, test accuracy
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=10000))
])

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred)
print(f"logisticregression test accuracy: {accuracy_lr:.4f}")
print()

# 3. RandomForest baseline (from lesson 11)
# your code — fit with n_estimators=100, random_state=42, test accuracy
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"RandomForestClassifer test accuracy: {accuracy_rf:.4f}")
print()

# 4. XGBClassifier with defaults (but n_estimators=100, random_state=42)
# Note: XGBoost doesn't need scaling (tree-based), so pass raw X
# your code — fit, test accuracy
# NOTE: If you get a warning about use_label_encoder, ignore it or pass
#       use_label_encoder=False
model_xgb = XGBClassifier(n_estimators=100, random_state=42)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBClassifier test accuracy: {accuracy_xgb:.4f}")

# 5. XGBClassifier with early stopping
# n_estimators=1000, learning_rate=0.05, early_stopping_rounds=20
# Fit with eval_set=[(X_test, y_test)]
# Print how many trees it actually used (model.best_iteration_)
# Print test accuracyprint("===== XGBoost (early stopping) =====")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)
model_xgb_early = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=20,
    random_state=42
)
model_xgb_early.fit(X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False
          )

print(f"No.of trees used: {model_xgb_early.best_iteration}")
y_pred_xgb_early = model_xgb_early.predict(X_test)
accuracy_xgb_early = accuracy_score(y_test, y_pred_xgb_early)
print(f"XGBClassifier with early stopping rounds: {accuracy_xgb_early:.4f}")
print()

# 6. LightGBM with similar settings
# LGBMClassifier(n_estimators=1000, learning_rate=0.05)
# Use callbacks for early stopping (LightGBM API is slightly different)
# Print test accuracy
model_lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42, verbose=-1)

model_lgbm.fit(X_tr, y_tr,
               eval_set=[(X_val, y_val)],
               callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
)

print(f"Trees actually used: {model_lgbm.best_iteration_}")
accuracy_lgbm = accuracy_score(y_test, model_lgbm.predict(X_test))
print(f"LightGBM test accuracy: {accuracy_lgbm:.4f}")
print()



# 7. Learning rate sweep for XGBoost:
# for lr in [0.01, 0.05, 0.1, 0.3]:
#     train XGB with n_estimators=1000, early stopping
#     print lr, best_iteration_, test accuracy
# Which learning rate finds the best result?
for lr in [0.01, 0.05, 0.1, 0.3]:
    model = XGBClassifier(n_estimators=1000, learning_rate=lr, early_stopping_rounds=20, random_state=42)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False
            )
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"lr={lr:<5} best_iter={model.best_iteration:<5} test_acc={acc:.4f}")
print()

# 8. Compare all models in a table:
# LogisticRegression | RandomForest | XGBoost | LightGBM
# Test accuracy for each
headers = ['model', 'test_acc']
content = [
    ["LogisticRegression", f"{accuracy_lr:.4f}"],
    ["RandomForest",       f"{accuracy_rf:.4f}"],
    ["XGBoost (default)",  f"{accuracy_xgb:.4f}"],
    ["XGBoost (early)",    f"{accuracy_xgb_early:.4f}"],
    ["LightGBM (early)",   f"{accuracy_lgbm:.4f}"],
]


print(tabulate(content, headers=headers, tablefmt="grid"))

         
