import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)

# ---------------------------------------------------------------
# 1. Load and split
# ---------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
# Class 0 = malignant, Class 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 2. Train logistic regression
# ---------------------------------------------------------------
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)                     # class labels (0 or 1)
y_proba = model.predict_proba(X_test)[:, 1]         # P(class 1) for each sample

# ---------------------------------------------------------------
# 3. Confusion matrix
# ---------------------------------------------------------------
# Convention in sklearn: [[TN, FP], [FN, TP]] with pos_label = 1 (default)
# Since class 1 = benign here, "positive" means BENIGN by default.
#   TN = actual malignant, predicted malignant
#   FP = actual malignant, predicted benign   -> MISSED CANCER
#   FN = actual benign,    predicted malignant -> false alarm
#   TP = actual benign,    predicted benign
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("  [[TN, FP],")
print("   [FN, TP]]  with default pos_label=1 (benign)")
print()

# ---------------------------------------------------------------
# 4. Metrics with DEFAULT pos_label (benign as positive)
# ---------------------------------------------------------------
# sklearn defaults pos_label=1 (the class labeled 1) — NOT the majority class.
# Here class 1 happens to be benign.
accuracy_b  = accuracy_score(y_test, y_pred)
precision_b = precision_score(y_test, y_pred)
recall_b    = recall_score(y_test, y_pred)
f1_b        = f1_score(y_test, y_pred)

print("=== Metrics with pos_label=1 (benign) ===")
print(f"accuracy:  {accuracy_b:.4f}")
print(f"precision: {precision_b:.4f}")
print(f"recall:    {recall_b:.4f}")
print(f"f1:        {f1_b:.4f}")
print()

# ---------------------------------------------------------------
# 5. ROC-AUC
# ---------------------------------------------------------------
# Uses PROBABILITIES, not classes.
# 0.5 = random guessing, 1.0 = perfect ranking.
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.4f}")
print()

# ---------------------------------------------------------------
# 6. Metrics with MALIGNANT as positive (pos_label=0)
# ---------------------------------------------------------------
# This is the medical framing: "positive" = has cancer.
# With pos_label=0:
#   FP = predicted malignant, actually benign    -> false alarm
#   FN = predicted benign,    actually malignant -> MISSED CANCER
precision_m = precision_score(y_test, y_pred, pos_label=0)
recall_m    = recall_score(y_test, y_pred, pos_label=0)
f1_m        = f1_score(y_test, y_pred, pos_label=0)

print("=== Metrics with pos_label=0 (malignant) ===")
print(f"precision: {precision_m:.4f}")
print(f"recall:    {recall_m:.4f}")
print(f"f1:        {f1_m:.4f}")
print()

# ---------------------------------------------------------------
# 7. Threshold sweep
# ---------------------------------------------------------------
# Lowering the threshold → more predicted positives → higher recall,
#                          lower precision.
# Raising the threshold  → fewer predicted positives → higher precision,
#                          lower recall.
# On this dataset the model is very confident (probabilities near 0 or 1),
# so 0.2–0.5 gives identical results. Only at higher thresholds do
# predictions start to shift.
for t in [0.2, 0.3, 0.5, 0.7]:
    y_pred_t = (y_proba >= t).astype(int)

    p = precision_score(y_test, y_pred_t, pos_label=0)
    r = recall_score(y_test, y_pred_t, pos_label=0)
    f = f1_score(y_test, y_pred_t, pos_label=0)

    print(f"threshold={t}: precision={p:.4f}, recall={r:.4f}, f1={f:.4f}")

print()

# ---------------------------------------------------------------
# 8. Full classification report (default pos_label=1)
# ---------------------------------------------------------------
print("Classification report (default pos_label=1):")
print(classification_report(y_test, y_pred, target_names=data.target_names))