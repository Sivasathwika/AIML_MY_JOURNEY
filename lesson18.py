import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tabulate import tabulate

#1. Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values per column:")
print(df.isna().sum())
print("\nData types:")
print(df.dtypes)

# 2. Pick useful columns
# Feature candidates: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
# Target: Survived
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
cat_cols = ['Pclass', 'Sex', 'Embarked']

X = df[features]
y = df[target]

# 3. Split into train/test FIRST (before any preprocessing)
# your code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build a ColumnTransformer:
#    - numeric pipeline: impute median + StandardScaler
#    - categorical pipeline: impute most_frequent + OneHotEncoder
#    Numerical columns: ['Age', 'SibSp', 'Parch', 'Fare']
#    Categorical columns: ['Pclass', 'Sex', 'Embarked']

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('Scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 5. Build full pipeline with LogisticRegression
#    Fit on train, predict on test, print accuracy
full_pipeline_lr = Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression(max_iter=10000))
])

full_pipeline_lr.fit(X_train, y_train)
acc_lr = accuracy_score(y_test, full_pipeline_lr.predict(X_test))

# 6. Try the same with RandomForestClassifier
#    (RF doesn't need scaling, but the pipeline is fine)
full_pipeline_rf = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])
full_pipeline_rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, full_pipeline_rf.predict(X_test))


print(f"Baseline LR accuracy: {acc_lr:.4f}")
print(f"Baseline RF accuracy: {acc_rf:.4f}")
print()

# 7. Feature engineering experiment:
#    Add two new features to the raw dataframe BEFORE the split:
#      FamilySize = SibSp + Parch + 1
#      IsAlone = (FamilySize == 1).astype(int)
#    Also extract Title from Name (e.g., "Mr", "Mrs", "Miss"):
#      df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
#    Then add FamilySize, IsAlone, Title to your feature set.
#    Re-run both models. Did accuracy improve?
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

#Group rare titles (<10 occurences) into 'Rare'
title_counts = df['Title'].value_counts()
rare_titles = title_counts[title_counts < 10].index
df['Title'] = df['Title'].replace(rare_titles, "Rare")

print("Title counts after grouping:")
print(df['Title'].value_counts())
print()

# splitting again with engineered features
features_eng = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'FamilySize', 'IsAlone', 'Title']
num_cols_e = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
cat_cols_e = ['Pclass', 'Sex', 'Embarked', 'Title']

X_new = df[features_eng]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

num_pipe_e = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler()),
])
cat_pipe_e = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore')),
])

prep_e = ColumnTransformer([
    ('num', num_pipe_e, num_cols_e),
    ('cat', cat_pipe_e, cat_cols_e),
])

pipe_lr_e = Pipeline([('prep', prep_e), ('model', LogisticRegression(max_iter=10000))])
pipe_rf_e = Pipeline([('prep', prep_e), ('model', RandomForestClassifier(random_state=42))])

pipe_lr_e.fit(X_train, y_train)
acc_lr_e = accuracy_score(y_test, pipe_lr_e.predict(X_test))

pipe_rf_e.fit(X_train, y_train)
acc_rf_e = accuracy_score(y_test, pipe_rf_e.predict(X_test))

print(f"Engineered LR accuracy: {acc_lr_e:.4f}")
print(f"Engineered RF accuracy: {acc_rf_e:.4f}")
print()

# 8. Print comparison table:
#    Model | Features | Test Accuracy
headers = ["Model", "Features", "Test Accuracy"]

content = [
     ["LogisticRegression", "Baseline",   f"{acc_lr:.4f}"],
    ["LogisticRegression", "Engineered", f"{acc_lr_e:.4f}"],
    ["RandomForest",       "Baseline",   f"{acc_rf:.4f}"],
    ["RandomForest",       "Engineered", f"{acc_rf_e:.4f}"],
]

print(tabulate(content, headers=headers, tablefmt="grid"))