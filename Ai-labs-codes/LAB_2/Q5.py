# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

# -------------------------------
# Step 1: Create toy dataset
# -------------------------------
data = {
    'Hours_Study': [1, 2, 3, 4, 5, 6, 7, 8],
    'Exam_Score': [40, 50, 55, 65, 70, 75, 85, 90],
    'Pass': [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

X = df[['Hours_Study']]
y_score = df['Exam_Score']  # For Linear Regression
y_pass = df['Pass']         # For Logistic Regression

# -------------------------------
# Step 2: Linear Regression
# -------------------------------
lin_model = LinearRegression()
lin_model.fit(X, y_score)
pred_scores = lin_model.predict(X)

print("Linear Regression Predictions (Exam Scores):")
for i, pred in enumerate(pred_scores):
    print(f"Hours Study: {X['Hours_Study'][i]}, Predicted Score: {pred:.2f}, Actual Score: {y_score[i]}")

# -------------------------------
# Step 3: Logistic Regression
# -------------------------------
log_model = LogisticRegression()
log_model.fit(X, y_pass)
pred_pass = log_model.predict(X)

print("\nLogistic Regression Predictions (Pass/Fail):")
for i, pred in enumerate(pred_pass):
    print(f"Hours Study: {X['Hours_Study'][i]}, Predicted Pass: {pred}, Actual Pass: {y_pass[i]}")

# -------------------------------
# Step 4: Comparison / Explanation
# -------------------------------
print("\nComparison & Explanation:")
print("""
- Linear Regression predicts continuous values (Exam Scores) and can produce values outside [0,1].
- Logistic Regression predicts probabilities and outputs 0 or 1 for classification (Pass/Fail).
- Using Linear Regression for classification is unsuitable because:
    1. Predictions can be less than 0 or greater than 1.
    2. Probabilities are not bounded.
    3. It does not model the S-shaped probability curve which Logistic Regression does.
- Logistic Regression is designed specifically for binary classification.
""")
