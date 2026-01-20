# AND Gate Neural Network using scikit-learn
import numpy as np
from sklearn.neural_network import MLPClassifier

# 1. Create Dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([0, 0, 0, 1])

# 2. Build Neural Network
# - hidden_layer_sizes=(2,) -> 1 hidden layer with 2 neurons
# - activation='relu' -> ReLU activation for hidden layer
# - solver='adam' -> optimizer
model = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', solver='adam', max_iter=1000, random_state=42)

# 3. Train the model
model.fit(X, y)

# 4. Evaluate Accuracy
accuracy = model.score(X, y)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# 5. Predictions
predictions = model.predict(X)

# 6. Compare predictions with actual output
print("\nInput | Actual | Predicted")
for i in range(len(X)):
    print(f"{X[i]} | {y[i]}      | {predictions[i]}")
