import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

X = np.array([
    [2, 60], [3, 65], [1, 50], [4, 70], [5, 80],
    [6, 85], [7, 90], [8, 95], [9, 92], [10, 98],
    [2, 55], [3, 60], [4, 75], [6, 88], [7, 91]
])

y = np.array([0,0,0,0,1,1,1,1,1,1,0,0,1,1,1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
shallow_model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

shallow_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
shallow_model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=0)
shallow_loss, shallow_acc = shallow_model.evaluate(X_test, y_test, verbose=0)
deep_model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

deep_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
deep_model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=0)
deep_loss, deep_acc = deep_model.evaluate(X_test, y_test, verbose=0)
print("Shallow ANN Accuracy:", round(shallow_acc * 100, 2), "%")
print("Deep Neural Network Accuracy:", round(deep_acc * 100, 2), "%")
