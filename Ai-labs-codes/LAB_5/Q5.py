# Multi-Class Classification: Iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target  # 0 = Setosa, 1 = Versicolor, 2 = Virginica

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature scaling (important for NN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build Neural Network
model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=5000, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Iris Classification Accuracy: {accuracy*100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
