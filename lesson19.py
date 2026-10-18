import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tabulate import tabulate


# ---------------------------------------------------------------
# 1. Load + split
# ---------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print()


# ---------------------------------------------------------------
# 2. Depth sweep — bias-variance tradeoff in action
# ---------------------------------------------------------------
print("=" * 60)
print("Depth sweep: DecisionTreeClassifier")
print("=" * 60)

depths = [1, 2, 3, 5, 10, 20, None]
rows = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    gap       = train_acc - test_acc

    # Classify: underfit / good / overfit / both
    if train_acc < 0.90 and gap < 0.05:
        diagnosis = "Underfit (high bias)"
    elif train_acc < 0.90 and gap >= 0.05:
        diagnosis = "Both (bias + variance)"
    elif gap < 0.03:
        diagnosis = "Good balance"
    elif gap < 0.06:
        diagnosis = "Slight overfit"
    else:
        diagnosis = "Overfit (high variance)"

    rows.append([str(depth), f"{train_acc:.4f}",
                 f"{test_acc:.4f}", f"{gap:.4f}", diagnosis])

print(tabulate(rows,
               headers=["max_depth", "Train Acc", "Test Acc", "Gap", "Diagnosis"],
               tablefmt="grid"))
print()


# ---------------------------------------------------------------
# 3. Which depth is best?
# ---------------------------------------------------------------
best_row = max(rows, key=lambda r: float(r[2]))
print(f"Best test accuracy at max_depth={best_row[0]}: {best_row[2]}")
print(f"At max_depth=None the gap is {rows[-1][3]} — that's the variance cost.")
print()


# ---------------------------------------------------------------
# 4. Learning curve — high-bias model (depth=1)
# ---------------------------------------------------------------
print("=" * 60)
print("Learning curve: DecisionTree(max_depth=1) — high bias")
print("=" * 60)

train_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
bias_rows = []

for frac in train_fractions:
    n = int(len(X_train) * frac)
    X_sub, y_sub = X_train[:n], y_train[:n]

    model = DecisionTreeClassifier(max_depth=1, random_state=42)
    model.fit(X_sub, y_sub)

    train_acc = accuracy_score(y_sub, model.predict(X_sub))
    test_acc  = accuracy_score(y_test, model.predict(X_test))
    bias_rows.append([f"{frac:.2f}", n,
                      f"{train_acc:.4f}", f"{test_acc:.4f}"])

print(tabulate(bias_rows,
               headers=["Fraction", "N samples", "Train Acc", "Test Acc"],
               tablefmt="grid"))
print("→ Both curves stay low. Train and test converge to a HIGH value.")
print("→ More data won't help. Need a more complex model or better features.\n")


# ---------------------------------------------------------------
# 5. Learning curve — high-variance model (depth=None)
# ---------------------------------------------------------------
print("=" * 60)
print("Learning curve: DecisionTree(max_depth=None) — high variance")
print("=" * 60)

var_rows = []

for frac in train_fractions:
    n = int(len(X_train) * frac)
    X_sub, y_sub = X_train[:n], y_train[:n]

    model = DecisionTreeClassifier(max_depth=None, random_state=42)
    model.fit(X_sub, y_sub)

    train_acc = accuracy_score(y_sub, model.predict(X_sub))
    test_acc  = accuracy_score(y_test, model.predict(X_test))
    gap       = train_acc - test_acc
    var_rows.append([f"{frac:.2f}", n,
                     f"{train_acc:.4f}", f"{test_acc:.4f}", f"{gap:.4f}"])

print(tabulate(var_rows,
               headers=["Fraction", "N samples", "Train Acc", "Test Acc", "Gap"],
               tablefmt="grid"))
print("→ Train accuracy stays near 1.0, test accuracy lower with a big gap.")
print("→ More data helps close the gap. Regularization would too.\n")


# ---------------------------------------------------------------
# 6. Learning curve — LogisticRegression (low-variance baseline)
# ---------------------------------------------------------------
print("=" * 60)
print("Learning curve: LogisticRegression (balanced)")
print("=" * 60)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

lr_rows = []

for frac in train_fractions:
    n = int(len(X_train) * frac)
    X_sub, y_sub = X_train[:n], y_train[:n]

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('model',  LogisticRegression(max_iter=10000))])
    pipe.fit(X_sub, y_sub)

    train_acc = accuracy_score(y_sub, pipe.predict(X_sub))
    test_acc  = accuracy_score(y_test, pipe.predict(X_test))
    gap       = train_acc - test_acc
    lr_rows.append([f"{frac:.2f}", n,
                    f"{train_acc:.4f}", f"{test_acc:.4f}", f"{gap:.4f}"])

print(tabulate(lr_rows,
               headers=["Fraction", "N samples", "Train Acc", "Test Acc", "Gap"],
               tablefmt="grid"))
print("→ Train and test stay close — low variance.")
print("→ Small gap at every data size. Bias is the main source of error.\n")


# ---------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------
print("=" * 60)
print("Summary — what would you fix, per model?")
print("=" * 60)
print("depth=1 (high bias):   add features, deeper tree, less regularization")
print("depth=None (high var): more data, prune, ensemble, more regularization")
print("LogisticRegression:    balanced. Add non-linear features if needed.")