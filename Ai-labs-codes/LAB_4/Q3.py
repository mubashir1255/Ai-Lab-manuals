import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Create Dataset Directly
data = {
    'study_hours': [5, 2, 8, 1, 6, 3, 7, 4, 9, 2, 10, 8],
    'attendance': [80, 60, 90, 50, 85, 70, 88, 75, 95, 55, 98, 92],
    'marks': [70, 40, 85, 30, 78, 55, 82, 65, 90, 45, 95, 88],
    'result': ['Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Pass', 'Pass', 'Fail', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

# Step 2: Convert 'result' to numeric (Pass=1, Fail=0)
df['result'] = df['result'].map({'Pass': 1, 'Fail': 0})

# Step 3: Split Features and Target
X = df[['study_hours', 'attendance', 'marks']]
y = df['result']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 6: Predictions and Evaluation
y_pred = rf.predict(X_test)

print("Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Feature Importance
importance = pd.Series(rf.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importance)

# Step 8: Plot Feature Importance
importance.plot(kind='bar', color='skyblue', title='Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
