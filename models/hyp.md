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


# NEW DATA (NO RANDOM SEED SET CAUSE IM RETARD)

lstm 3 epochs 64 batch_size 500 review_max_length
test loss:     0.3380391001701355
test accuracy: 0.8695999979972839

lstm 3 epochs 64 batch_size 1000 review_max_length
test loss:     0.3424083888530731
test accuracy: 0.866100013256073

lstm 3 epochs 128 batch_size 1000 review_max_length
test loss:     0.33597591519355774
test accuracy: 0.8600999712944031

lstm 6 epochs 128 batch_size 1000 review_max_length
test loss:     0.3872539699077606
test accuracy: 0.8633000254631042

-----------------------------------------------

lstm 3 epochs 128 batch_size 500 review_max_length
test loss:     0.3661049008369446
test accuracy: 0.8708999752998352

lstm 3 epochs 128 batch_size 500 review_max_length
test loss:     0.32857751846313477
test accuracy: 0.8661999702453613

lstm 3 epochs 128 batch_size 500 review_max_length
test loss:     0.3426782786846161
test accuracy: 0.8572999835014343

-----------------------------------------------

lstm 4 epochs 128 batch_size 500 review_max_length
test loss:     0.35371142625808716
test accuracy: 0.8636000156402588

-----------------------------------------------

# WITH RANDOM SEED 1337

bidirectional LSTM 3 epochs 128 batch_size 500 review_max_length
test loss:     0.3311542868614197
test accuracy: 0.8651000261306763

LSTM 3 epochs 128 batch_size 500 review_max_length
test loss:     0.3715202808380127
test accuracy: 0.8341000080108643
