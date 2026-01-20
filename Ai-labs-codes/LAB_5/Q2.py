# Neural Network Regression: y = x^2 + noise
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 1. Generate dataset
np.random.seed(42)
X = np.random.uniform(-3, 3, 100).reshape(-1, 1)  # 100 samples
noise = np.random.normal(0, 1, X.shape)           # Gaussian noise
y = X**2 + noise                                   # y = x^2 + noise

# 2. Build Neural Network
# hidden_layer_sizes=(10,) -> 1 hidden layer with 10 neurons
model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=5000, random_state=42)
model.fit(X, y.ravel())  # y.ravel() converts (100,1) to (100,)

# 3. Predictions
y_pred = model.predict(X)

# 4. Plot Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='Actual')       # Actual points
plt.scatter(X, y_pred, color='red', label='Predicted') # Predicted points
plt.title("Neural Network Regression: y = x^2 + noise")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# 5. Experiment with different hidden neurons
neurons_list = [1, 5, 10, 50]
plt.figure(figsize=(12,8))

for i, neurons in enumerate(neurons_list, 1):
    model = MLPRegressor(hidden_layer_sizes=(neurons,), activation='relu', solver='adam', max_iter=5000, random_state=42)
    model.fit(X, y.ravel())
    
    # Smooth curve for plotting
    X_grid = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_grid_pred = model.predict(X_grid)
    
    plt.subplot(2, 2, i)
    plt.scatter(X, y, color='blue', label='Actual')
    plt.plot(X_grid, y_grid_pred, color='red', label=f'Predicted ({neurons} neurons)')
    plt.title(f'Hidden Neurons: {neurons}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()
