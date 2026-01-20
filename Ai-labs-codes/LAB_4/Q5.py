# Step 1: Import Libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Wine Dataset
wine = load_wine()
X = wine.data
y = wine.target

# Step 3: Split into Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Step 5: Train SVM Model (RBF Kernel)
svm_model = SVC(kernel='rbf', gamma='scale', C=1)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Step 6: Evaluate Both Models
rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)

print("Random Forest Accuracy:", round(rf_acc * 100, 2), "%")
print("SVM (RBF Kernel) Accuracy:", round(svm_acc * 100, 2), "%")

# Step 7: Classification Reports
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))

# Step 8: Conclusion
if rf_acc > svm_acc:
    print("\nConclusion: Random Forest performs better on this dataset.")
elif svm_acc > rf_acc:
    print("\nConclusion: SVM performs better on this dataset.")
else:
    print("\nConclusion: Both models perform equally well.")

# Step 9: Optional – Confusion Matrices
# Random Forest Confusion Matrix
plt.figure(figsize=(6,4))
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
