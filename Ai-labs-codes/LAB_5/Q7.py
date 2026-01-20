# Lightweight MNIST alternative using sklearn digits dataset
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load digits dataset (8x8 images, 0-9)
digits = load_digits()
X, y = digits.data, digits.target

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500, test_size=100, random_state=42, stratify=y)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build small MLP
model = MLPClassifier(hidden_layer_sizes=(32,), activation='relu',
                      solver='adam', alpha=0.001,
                      batch_size=16, max_iter=20, random_state=42, verbose=True)

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Digits Test Accuracy (tiny subset): {accuracy*100:.2f}%")

# 7. Plot first 10 predictions
plt.figure(figsize=(10,3))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[i].reshape(8,8), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
