import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate
from sklearn.tree import export_text

#1.Load + aplit (same as lesson 8/9)
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2.Train a DecisionTreeClassifier with default parameters
model_default = DecisionTreeClassifier()
model_default.fit(X_train, y_train)

#3.Predict, compute accuracy on both train and test
#4.Now train with max_depth=3 and re-evaluate
#5.Try max_depth=1, 2, 3, 5, 10, None
#for each, compute train accuracy and test accuracy
#print them in a table
depths = [1, 2, 3, 5, 10, None]
rows = []

for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, m.predict(X_train))
    test_acc  = accuracy_score(y_test,  m.predict(X_test))
    rows.append([str(d), train_acc, test_acc])

# add the default model as well (should equal None)
model_default = DecisionTreeClassifier(random_state=42)
model_default.fit(X_train, y_train)
rows.insert(0, ["default",
                accuracy_score(y_train, model_default.predict(X_train)),
                accuracy_score(y_test,  model_default.predict(X_test))])

print(tabulate(rows, headers=["max_depth", "Train Acc", "Test Acc"], tablefmt="grid"))

# For the best depth, print the classification report
# the best max_depth is 3 because it got the highest acc_train, acc_test and less difference compared to other depths 
best_model = DecisionTreeClassifier(max_depth=3, random_state=42)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print("Classification report (max_depth=3):")
print(classification_report(y_test, y_pred_best,
                            target_names=data.target_names))

#7.BONUS: visualize the tree with max_depth=3
#   from sklearn.tree import export_text
#   print(export_text(model, feature_names=list(data.feature_names)))
print("\nDecision tree (max_depth=3) as text rules:")
print(export_text(best_model, feature_names=list(data.feature_names)))


"""
# LESSON 10 — ANSWERS

# Q1. What is the default max_depth?
None (unlimited). Confirmed by the fixed run — `default` and `max_depth=None`
produce identical results (train=1.0, test=0.947).

# Q2. With default parameters, is there a gap between train and test?
Yes. Train = 1.0, test = 0.947. Gap = 0.053.
This means the model is OVERFITTING — it memorized the training data.
But test is still 0.947 because the dataset is clean and separable, so
even an overfit tree generalizes reasonably well. Overfitting is about
the GAP, not the absolute test score.

# Q3. With max_depth=1, underfitting or overfitting?
Underfitting. Train=0.921, test=0.895. Both are relatively low, and the
gap is small (0.026). The model is too simple — a single split can't
separate malignant from benign well. Underfitting = high train error
AND high test error.

# Q4. With max_depth=None, underfitting or overfitting?
Overfitting. Train=1.0, test=0.947. Large gap (0.053). The tree grew
until every training sample was perfectly classified, but this didn't
improve test performance. Overfitting = low train error AND high test
error (relative to train).

# Q5. Which depth gives the best test accuracy?
Depths 3, 5, 10, and None ALL give test=0.947. They tie on test
accuracy. But depth=3 is the best CHOICE because it achieves the same
test performance with the smallest gap (0.031) — least overfitting.
"Best" isn't just the highest test number. It's the simplest model
that achieves that number.

# Q6. Decision tree vs logistic regression?
- Logistic regression draws ONE straight line (linear boundary)
  between the classes. Fixed shape.
- Decision tree draws AXIS-ALIGNED RECTANGLES — a staircase of
  if/else rules. Can fit any shape.
- Logistic: predicts via `w·x + b`, thresholded.
- Tree: predicts by walking down a series of feature > threshold splits.
- Tree = non-linear and interpretable. Logistic = linear and stable.

# Q7. Why doesn't a decision tree need feature scaling?
A tree only compares each feature to a threshold ("is radius > 15?").
Rescaling changes the threshold value ("is radius/10 > 1.5?") but the
SAME samples go left and right. Order matters, magnitude doesn't.

Logistic regression uses w·x + b. If one feature has a range of
0.001 and another has 1000, their gradients are wildly different sizes
and gradient descent struggles. Scaling equalizes them.
"""
