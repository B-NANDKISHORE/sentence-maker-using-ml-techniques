def predict_gpt2(input_text, num_words):
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=len(input_ids[0]) + num_words, num_return_sequences=1)
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

# Flask App Setup
app = Flask(__name__)
