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
