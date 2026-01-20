# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -------------------------------
# Step 1: Create toy dataset
# -------------------------------
data = {
    'BMI': [22, 28, 30, 25, 27, 32, 24, 29],
    'Age': [25, 45, 50, 35, 40, 55, 30, 48],
    'Glucose': [90, 150, 160, 120, 140, 180, 100, 155],
    'Diabetic': [0, 1, 1, 0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Features and target
X = df[['BMI', 'Age', 'Glucose']]
y = df['Diabetic']

# -------------------------------
# Step 2: Fit Logistic Regression
# -------------------------------
model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# Step 3: Predict on training data (for metrics)
# -------------------------------
y_pred = model.predict(X)

# Compute metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# -------------------------------
# Step 4: Predict for new patient
# -------------------------------
new_patient = pd.DataFrame({'BMI':[28], 'Age':[45], 'Glucose':[150]})
prediction = model.predict(new_patient)[0]
probability = model.predict_proba(new_patient)[0][1]

print("\nPrediction for patient (BMI=28, Age=45, Glucose=150):")
print("Diabetic?" , "Yes" if prediction==1 else "No")
print("Probability of being Diabetic:", probability)
