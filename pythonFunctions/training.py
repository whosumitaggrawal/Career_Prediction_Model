# Importing Libraries
import os
import pickle
import pandas as pd
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Loading Dataset
df = pd.read_csv("data/mldata.csv")


# Number Encoding
cols = [
    "self-learning capability?",
    "Extra-courses did",
    "Taken inputs from seniors or elders",
    "worked in teams ever?",
    "Introvert",
]
for col in cols:
    df[col] = df[col].replace({"yes": 1, "no": 0})

mycol = ["reading and writing skills", "memory capability score"]
for col in mycol:
    df[col] = df[col].replace({"poor": 0, "medium": 1, "excellent": 2})


# Label Encoding
category_cols = [
    "certifications",
    "workshops",
    "Interested subjects",
    "interested career area ",
    "Type of company want to settle in?",
    "Interested Type of Books",
]
for col in category_cols:
    df[col] = df[col].astype("category")
    df[col + "_code"] = df[col].cat.codes


# Dummy Variable Encoding
df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])


# Building training frame
feed = df[
    [
        "Logical quotient rating",
        "coding skills rating",
        "hackathons",
        "public speaking points",
        "self-learning capability?",
        "Extra-courses did",
        "Taken inputs from seniors or elders",
        "worked in teams ever?",
        "Introvert",
        "reading and writing skills",
        "memory capability score",
        "B_hard worker",
        "B_smart worker",
        "A_Management",
        "A_Technical",
        "Interested subjects_code",
        "Interested Type of Books_code",
        "certifications_code",
        "workshops_code",
        "Type of company want to settle in?_code",
        "interested career area _code",
        "Suggested Job Role",
    ]
]


# Features/target split
x = feed.drop("Suggested Job Role", axis=1)
y = feed["Suggested Job Role"]


# Stratified train-test split for stable multi-class evaluation
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Decision Tree Classifier
clf1 = tree.DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=12,
    min_samples_leaf=2,
)
clf1.fit(x_train, y_train)

# SVM Classifier (scaling improves RBF SVM performance)
clf2 = make_pipeline(
    StandardScaler(),
    svm.SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=42),
)
clf2.fit(x_train, y_train)

# Random Forest Classifier
clf3 = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1,
)
clf3.fit(x_train, y_train)

# XGBoost Classifier
clf4 = XGBClassifier(
    random_state=42,
    learning_rate=0.05,
    n_estimators=400,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
)
clf4.fit(x_train, y_train)


# Evaluation summary
for name, model in {
    "DecisionTree": clf1,
    "SVM": clf2,
    "RandomForest": clf3,
    "XGBoost": clf4,
}.items():
    preds = model.predict(x_test)
    print(f"{name}: accuracy={accuracy_score(y_test, preds):.4f}, macro_f1={f1_score(y_test, preds, average='macro'):.4f}")


# Persist models
os.makedirs("pkl", exist_ok=True)

with open("pkl/model1.pkl", "wb") as file1:
    pickle.dump(clf1, file1)

with open("pkl/model2.pkl", "wb") as file2:
    pickle.dump(clf2, file2)

with open("pkl/model3.pkl", "wb") as file3:
    pickle.dump(clf3, file3)

with open("pkl/model4.pkl", "wb") as file4:
    pickle.dump(clf4, file4)

print("All Model Building Done!")
