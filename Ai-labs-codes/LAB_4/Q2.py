# Step 1: Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

cancer = load_breast_cancer()
X = cancer.data      # features
y = cancer.target    # target labels (0 = malignant, 1 = benign)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = SVC(kernel='linear')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
