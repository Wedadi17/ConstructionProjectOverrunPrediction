import joblib, pandas as pd, streamlit as st


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


def predict_overrun(X, threshold):
    proba = clf.predict_proba(X)[0, 1]
    pred = int(proba >= threshold)
    return proba, pred


st.title("📊 Construction Project Overrun Predictor")

st.info(
    "This app predicts whether a project will overrun (Yes/No). "
    "Budget deviation prediction was removed because the regression model was not reliable (R² ≤ 0) "
    "with the available dataset variables."
)

threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)

budget = st.number_input("Project Budget Amount", min_value=0.0, value=100000.0, step=1000.0)
duration = st.number_input("Project Duration (Days)", min_value=0.0, value=12.0, step=1.0)
phase = st.selectbox("Project Phase Name", ["Scope", "Design", "Construction"])

if st.button("Predict"):
    X_input = prepare_input(budget, duration, phase)
    proba, pred = predict_overrun(X_input, threshold)

    st.subheader("Result")
    st.metric("Overrun Probability", f"{proba:.2%}")
    st.metric("Prediction (0=No, 1=Yes)", pred)

    if pred == 1:
        st.success("Likely to overrun.")
    else:
        st.warning("Likely NOT to overrun.")
