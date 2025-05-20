# Step 3: Build the models
# LSTM Model
lstm_model = Sequential()
lstm_model.add(Embedding(total_words, 100))
lstm_model.add(LSTM(150, return_sequences=True))
lstm_model.add(LSTM(150))
lstm_model.add(Dense(total_words, activation='softmax'))

# Bi-directional LSTM Model
bi_lstm_model = Sequential()
bi_lstm_model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
bi_lstm_model.add(Bidirectional(LSTM(150, return_sequences=True)))
bi_lstm_model.add(Bidirectional(LSTM(150)))
bi_lstm_model.add(Dense(total_words, activation='softmax'))

# GRU Model
gru_model = Sequential()
gru_model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
gru_model.add(GRU(150, return_sequences=True))
gru_model.add(GRU(150))
gru_model.add(Dense(total_words, activation='softmax'))

# RNN Model
rnn_model = Sequential()
rnn_model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
rnn_model.add(SimpleRNN(150, return_sequences=True))
rnn_model.add(SimpleRNN(150))
rnn_model.add(Dense(total_words, activation='softmax'))

