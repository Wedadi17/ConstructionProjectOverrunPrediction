import joblib, pandas as pd

clf = joblib.load("artifacts/clf.joblib")
cols = joblib.load("artifacts/cols.joblib")


def prepare_input(budget, duration, phase):
    X = pd.DataFrame([{
        "Project Budget Amount": budget,
        "Project Duration": duration,
        "Project Phase Name": phase
    }])
    X = pd.get_dummies(X, columns=["Project Phase Name"], drop_first=True)
    X = X.reindex(columns=cols, fill_value=0)
    return X


def predict_overrun(X, threshold=0.5):
    proba = clf.predict_proba(X)[0, 1]
    pred = int(proba >= threshold)
    return proba, pred


# ===== Example Run =====
budget = 200000
duration = 12
phase = "Construction"

X_input = prepare_input(budget, duration, phase)
proba, pred = predict_overrun(X_input, threshold=0.5)

print("\n--- Overrun Prediction ---")
print("Probability of overrun:", round(proba, 4))
print("Prediction (0=No, 1=Yes):", pred)
