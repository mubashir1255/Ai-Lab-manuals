def predict_next_word(model, tokenizer, text, max_len):
    token_text = tokenizer.texts_to_sequences([text])[0]
    token_text = pad_sequences([token_text], maxlen=max_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_text, verbose=0))

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

user_input = input("Enter a sentence: ")
next_word = predict_next_word(model, tokenizer, user_input, max_len)
print("Next Word Prediction:", next_word)
