import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Sample stock price data
data = pd.DataFrame({"Price": np.linspace(100, 200, 300)})

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

model = Sequential([
    SimpleRNN(50, input_shape=(60,1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, verbose=0)

predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)

plt.plot(data.values, label="Actual")
plt.plot(range(60, len(predicted)+60), predicted, label="Predicted")
plt.legend()
plt.show()
