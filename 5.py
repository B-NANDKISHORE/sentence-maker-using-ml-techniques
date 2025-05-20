def predict_next_words(model, tokenizer, input_text, num_words=5):
    input_text = input_text.lower()
    generated_words = []
    
    # Generate words iteratively
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        
        # Get the index of the predicted word
        predicted_word_index = np.argmax(predicted)
        
        # Find the word corresponding to the predicted index
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        
        # Append predicted word and update input_text for next prediction
        if output_word:
            generated_words.append(output_word)
            input_text += " " + output_word
        else:
            break
    
    return " ".join(generated_words)

