import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


faqs = """About the Program
What is the course fee for Data Science Mentorship Program (DSMP 2023)
The course follows a monthly subscription model where you have to make monthly payments of Rs 799/month.
What is the total duration of the course?
The total duration of the course is 7 months. So the total course fee becomes 799*7 = Rs 5600(approx.)
What is the syllabus of the mentorship program?
We will be covering the following modules:
Python Fundamentals
Python libraries for Data Science
Data Analysis
SQL for Data Science
Maths for Machine Learning
ML Algorithms
Practical ML
MLOPs
Case studies"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])

input_sequences = []
for line in faqs.split('\n'):
    tokenized_line = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokenized_line)):
        input_sequences.append(tokenized_line[:i+1])

# Maximum sequence length
max_len = max([len(x) for x in input_sequences])

# Pad sequences
padded_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]
y = to_categorical(y, num_classes=len(tokenizer.word_index)+1)

# -----------------------------
# Build LSTM Model
# -----------------------------
vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(vocab_size, 50, input_length=max_len-1),
    LSTM(150, return_sequences=True),
    LSTM(150),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# -----------------------------
# Train Model
# -----------------------------
print("Training LSTM model for Question 3...")
model.fit(X, y, epochs=100, verbose=1)

# -----------------------------
# Function to predict next word
# -----------------------------
def predict_next_word(model, tokenizer, text, max_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted_index = np.argmax(model.predict(token_list, verbose=0))
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# -----------------------------
# Function to generate 5 sequential words
# -----------------------------
def generate_5_words(seed_text, model, tokenizer, max_len):
    generated_text = seed_text
    for _ in range(5):
        next_word = predict_next_word(model, tokenizer, generated_text, max_len)
        generated_text += " " + next_word
    return generated_text

# -----------------------------
# Example usage
# -----------------------------
seed_text = input("Enter seed text to generate 5 words: ")
generated_text = generate_5_words(seed_text, model, tokenizer, max_len)
print("\nGenerated Text (5 words):")
print(generated_text)
