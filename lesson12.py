import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#1.Load + split
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=====Baseline=====")
#2. Baseline - logisticregression on UNSCALED features
#max_iter=10000. Fit, predict, print test accuracy
model_u = LogisticRegression(max_iter=10000)
model_u.fit(X_train, y_train)
y_pred_u = model_u.predict(X_test)
acc_u = accuracy_score(y_test, y_pred_u)
print(f"Baseline accuracy using logistic regression on uncaled features: {acc_u:0.4f}\n")


print("=====StandardScaler=====")
#3.Now scale with StandardScaler
# fit on X_train only
#Transform both X_train and X_test
#Fit logistic regression on the scaled data
#print test accuracy
#Did it improve?
st_scaler = StandardScaler()
st_scaler.fit(X_train)
X_train_scaled = st_scaler.transform(X_train)
X_test_scaled = st_scaler.transform(X_test)

model_st = LogisticRegression(max_iter=10000)
model_st.fit(X_train_scaled, y_train)
y_pred_st = model_st.predict(X_test_scaled)
acc_st = accuracy_score(y_test, y_pred_st)
print(f"Accuracy using logistic regression on scaled features: {acc_st:0.4f}")
print(f"StandardScaler improved accuracy by {acc_st - acc_u:0.4f}\n" if acc_st > acc_u else f"StandardScaler did not improve accuracy, it decreased by {acc_u - acc_st:0.4f}\n")


print("=====MinMaxScaler=====")
#4.Same with MinMaxScaler
#print test acccuracy
#which scaler wins? by how much?
mms_scaler = MinMaxScaler()
mms_scaler.fit(X_train)
X_train_mms = mms_scaler.transform(X_train)
X_test_mms = mms_scaler.transform(X_test)

model_mms = LogisticRegression(max_iter=10000)
model_mms.fit(X_train_mms, y_train)

y_pred_mms = model_mms.predict(X_test_mms)
acc_mms = accuracy_score(y_test, y_pred_mms)

print(f"Accuracy using logistic regression on MinMax scaled features: {acc_mms:0.4f}")
print(f"MinMaxScaler improved accuracy by {acc_mms - acc_u:0.4f}\n" if acc_mms > acc_u else f"MinMaxScaler did not improve accuracy, it decreased by {acc_u - acc_mms:0.4f}\n")
print(f"StandardScaler wins by {acc_st - acc_mms:0.4f}" if acc_st > acc_mms else f"MinMaxScaler wins by {acc_mms - acc_st:0.4f}\n")


print("=====Pipeline=====")
#5. Buil a Pipeline for Standardscaler _ LogisticRegression
#fit on X_train, predict on X_test
#print accuracy
#compare to step 3 - should be identical
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=10000))
])

pipe.fit(X_train, y_train)
y_pred_pipe_st = pipe.predict(X_test)
acc_pipe_st = accuracy_score(y_test, y_pred_pipe_st)
print(f"Accuracy using pipeline with StandardScaler and LogisticRegression: {acc_pipe_st:0.4f}")
print(f"Pipeline accuracy is identical to standard scaler without pipeline: {acc_pipe_st == acc_st}\n"
       if acc_pipe_st == acc_st 
       else f"Pipeline accuracy is not identical to standard scaler without pipeline: {acc_pipe_st != acc_st}\n")


print("=====Cross-validation without pipeline (leaky)=====")
#6.Cross-validate without pipeline (leaky):
# Scale all X with Standard scaler (fit_transform on all X)
# cross_val_score(logistic regression, X_scaled, y, cv=5)
# Print mean ± std
X_scaled_all = st_scaler.transform(X)
cv_scores_leaky = cross_val_score(LogisticRegression(max_iter=10000), X_scaled_all, y, cv=5)
print(f"Cross-validation scores: {cv_scores_leaky}")
print(f"Mean ± std of cross-validation scores: {cv_scores_leaky.mean():0.6f} ± {cv_scores_leaky.std():0.6f}\n")


print("=====Cross-validation with pipeline (correct)=====")
#7. Cross-validate with a pipeline (correct):
# cross_val_score(pipeline, X, y, cv=5)
#print mean ± std
#Are the means different? which one is lying and why?
cv_scores_correct = cross_val_score(pipe, X, y, cv=5)
print(f"Cross validation scores :{cv_scores_correct}")
print(f"Mean ± std of cross-validation scores: {cv_scores_correct.mean():0.6f} ± {cv_scores_correct.std():0.6f}\n")


print(f"\n=====Random Forest Classifier=====")
#8. Show that scaling does not help tree models:
# RandomForestClassiferon unscaled X, cross_val_score
# Random_ForestClassifier on scaled X, cross_val_score
# Print both means
# Are they different?
cv_scores_rf_unscaled = cross_val_score(RandomForestClassifier(max_depth=5, random_state=42), X, y, cv=5)
print(f"Cross-validation scores on unscaled features: {cv_scores_rf_unscaled}")
print(f"Mean cross-validation score (unscaled): {cv_scores_rf_unscaled.mean():0.4f}")

pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(max_depth=5, random_state=42))
])

cv_scores_rf_scaled = cross_val_score(pipe_rf, X, y, cv=5)
print(f"Cross-validation scores on scaled features: {cv_scores_rf_scaled}")
print(f"Mean cross-validation score (scaled): {cv_scores_rf_scaled.mean():0.4f}")

