old_LSTM.keras

# Build the LSTM model
model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length),
  LSTM(lstm_units, return_sequences=False),
  Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['F1Score'])

Final Loss: 0.3670, Final Accuracy: 0.6648


# NEW DATA

lstm 3 epochs 64 batch_size 500 review_max_length
test loss:     0.3380391001701355
test accuracy: 0.8695999979972839

lstm 3 epochs 64 batch_size 1000 review_max_length
test loss:     0.3424083888530731
test lccuracy: 0.866100013256073
