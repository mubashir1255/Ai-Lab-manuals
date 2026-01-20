from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import time

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

def train_model(layers, neurons, batch):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    
    for _ in range(layers):
        model.add(Dense(neurons, activation='relu'))
    
    model.add(Dense(10, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start = time.time()
    model.fit(X_train, y_train, epochs=5, batch_size=batch, verbose=0)
    end = time.time()
    
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    return acc, end-start

configs = [
    (2, 64, 32),
    (3, 128, 32),
    (4, 256, 64)
]

for c in configs:
    acc, t = train_model(*c)
    print(f"Layers:{c[0]}, Neurons:{c[1]}, Batch:{c[2]} -> Accuracy:{acc:.3f}, Time:{t:.2f}s")
