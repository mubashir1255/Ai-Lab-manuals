import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Simple music notes (encoded)
notes = [60, 62, 64, 65, 67, 69, 71, 72]

X, y = [], []
for i in range(3, len(notes)):
    X.append(notes[i-3:i])
    y.append(notes[i])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    SimpleRNN(50, input_shape=(3,1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

predicted_note = model.predict(X[:1])
print("Generated Note:", int(predicted_note[0][0]))
