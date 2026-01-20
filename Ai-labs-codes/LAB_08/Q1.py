import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

faqs = """what is the course fee
the course fee is monthly
what is the total duration
the total duration is seven months"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])

input_sequences = []
for line in faqs.split('\n'):
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        input_sequences.append(tokens[:i+1])

max_len = max(len(x) for x in input_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

model = Sequential([
    Embedding(len(tokenizer.word_index)+1, 50, input_length=max_len-1),
    LSTM(100),
    Dense(len(tokenizer.word_index)+1, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, epochs=200, verbose=0)

# -----------------------------
# Predict Next Word
# -----------------------------
input_text = "what is the"
token_text = tokenizer.texts_to_sequences([input_text])[0]
token_text = pad_sequences([token_text], maxlen=max_len-1, padding='pre')

predicted_index = np.argmax(model.predict(token_text, verbose=0))

for word, index in tokenizer.word_index.items():
    if index == predicted_index:
        print("Input:", input_text)
        print("Predicted Next Word:", word)
