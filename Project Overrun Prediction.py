
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import false
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Dataset file reading
df = pd.read_csv("Capital_Project_Schedules_and_Budgets.csv")

# Renaming the variables
df.columns = ['Project_Phase','Baseline_Budget','Estimate_At_Completion'
              ,'Overrun_Status','Budget_Overrun/Saved_Amount']

# prediction if the project will be overrun

#Dymming
df = pd.get_dummies(df, columns=['Project_Phase'], drop_first=True)
# Features
X = df.drop(['Overrun_Status','Budget_Overrun/Saved_Amount'], axis=1)

# Target
y_class = df['Overrun_Status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluation
print(classification_report(y_test, clf.predict(X_test)))

# Saving the model and feature structure

feature_cols = X.columns
joblib.dump(clf, "clf_model.joblib")
joblib.dump(feature_cols, "feature_cols.joblib")

# Reducing the MAE error

reg = RandomForestRegressor(n_estimators=300,max_depth=10, random_state=42)

# Predicting the overrun budget amount

df_overrun = df[df['Overrun_Status'] == 1]

X_reg = df_overrun.drop(['Overrun_Status','Budget_Overrun/Saved_Amount'], axis=1)
y_reg = df_overrun['Budget_Overrun/Saved_Amount']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = RandomForestRegressor(random_state=42)
reg.fit(Xr_train, yr_train)

# Saving the regresion model and freature

print("MAE:", mean_absolute_error(yr_test, reg.predict(Xr_test)))

# checking error
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, clf.predict(X_test))

# prediction

# Raw input (human format)
new_raw = pd.DataFrame([{
    "Project_Phase": "Design",
    "Baseline_Budget": 10000,
    "Estimate_At_Completion": 50000
}])

new_project = pd.get_dummies(new_raw, columns=["Project_Phase"], drop_first=True)







# Example new project (change numbers to your real values)



# Make sure columns match training columns exactly

new_project = new_project.reindex(columns=feature_cols, fill_value=0)


# Predict class

print('######## this the prediction section ##################')
pred_class = clf.predict(new_project)[0]
print("Overrun prediction (0/1):", pred_class)

# Predict probability (more useful than only 0/1)
pred_proba = clf.predict_proba(new_project)[0, 1]
print("Probability of overrun:", pred_proba)

if pred_class == 1:
    pred_amount = reg.predict(new_project)[0]
    print("Predicted Budget_Overspent_Amount:", pred_amount )
else:
    print("No overrun predicted, regression not applied.")

# Saving the trained and Freature_cols

import joblib

joblib.dump(clf, "clf.joblib")
joblib.dump(feature_cols, "feature_cols.joblib")

