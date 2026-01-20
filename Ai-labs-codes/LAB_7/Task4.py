import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Fake temperature data
temps = np.sin(np.arange(0, 200)) * 10 + 20

X, y = [], []
for i in range(10, len(temps)):
    X.append(temps[i-10:i])
    y.append(temps[i])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    SimpleRNN(50, input_shape=(10,1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, verbose=0)

pred = model.predict(X)

plt.plot(temps[10:], label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.show()
