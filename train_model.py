import os, joblib, pandas as pd
git add requirements.txt.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score

DATA = "Capital_Project_Schedules_and_Budgets.csv"
OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

# Project Features / dependent variables
X = df[["Project Budget Amount", "Project Duration", "Project Phase Name"]]
X = pd.get_dummies(X, columns=["Project Phase Name"], drop_first=True)

# Project Target / independent variable
y = df["Overrun"]

# Remove missing values, ordering and cleaning the
data = pd.concat([X, y], axis=1).dropna()
X = data[X.columns]
y = data["Overrun"].astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
clf = RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced")

# Train
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Recall (Overrun=1):", recall_score(y_test, y_pred, pos_label=1))

# Save model + columns
joblib.dump(clf, f"{OUT}/clf.joblib")
joblib.dump(list(X.columns), f"{OUT}/cols.joblib")

print("\n✅ Overrun model saved to artifacts/")
