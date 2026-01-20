import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. XOR dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([0,1,1,0])  # XOR outputs

# 2. Activation functions to test
activations = ['logistic', 'tanh', 'relu']  # 'logistic' = sigmoid

# Store results
results = {}

for act in activations:
    # Build MLP with 1 hidden layer of 4 neurons (enough for XOR)
    model = MLPClassifier(hidden_layer_sizes=(4,), activation=act, solver='adam', max_iter=5000, random_state=42)
    
    # Train model
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Accuracy
    acc = accuracy_score(y, y_pred)
    
    # Loss at final iteration
    loss = model.loss_
    
    results[act] = {'accuracy': acc, 'loss': loss, 'predictions': y_pred, 'iterations': model.n_iter_}
    
    print(f"Activation: {act}")
    print(f"Predictions: {y_pred}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Final Loss: {loss:.4f}")
    print(f"Iterations to converge: {model.n_iter_}\n")

# 3. Plot comparison of predictions
plt.figure(figsize=(10,4))
for i, act in enumerate(activations, 1):
    plt.subplot(1, 3, i)
    plt.bar(['00','01','10','11'], results[act]['predictions'], color='orange')
    plt.title(f"Activation: {act}")
    plt.ylim(0,1.2)
    plt.ylabel("Predicted Output")
    plt.xlabel("XOR Input")
plt.tight_layout()
plt.show()

# 4. Compare Accuracy, Loss, and Iterations
plt.figure(figsize=(8,4))
plt.bar(activations, [results[a]['accuracy'] for a in activations], color='green')
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(8,4))
plt.bar(activations, [results[a]['loss'] for a in activations], color='red')
plt.title("Final Loss Comparison")
plt.ylabel("Loss")
plt.show()

plt.figure(figsize=(8,4))
plt.bar(activations, [results[a]['iterations'] for a in activations], color='blue')
plt.title("Convergence Speed (Iterations)")
plt.ylabel("Iterations")
plt.show()
