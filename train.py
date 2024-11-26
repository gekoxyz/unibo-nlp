import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.4

NUM_EPOCHS = 3
BATCH_SIZE = 64
REVIEW_MAX_LENGTH = 1000

hyperparameters = {
  "NUM_EPOCHS": [3, 6],
  "BATCH_SIZE": [64, 128],
  "REVIEW_MAX_LENGTH": [500, 1000]
}

TOKENIZER_MAX_WORDS = 10000

data = pd.read_csv("./lemmatized_clean_dataset.csv")

train_data, test_data = train_test_split(data, test_size=TEST_SPLIT, random_state=1337, stratify=data["label"])

print(f"train data shape: {train_data.shape}")
print(f"test data shape:  {test_data.shape}")

# tokenize text data
tokenizer = Tokenizer(num_words=TOKENIZER_MAX_WORDS)
tokenizer.fit_on_texts(train_data["text"])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["text"]), maxlen=REVIEW_MAX_LENGTH)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["text"]), maxlen=REVIEW_MAX_LENGTH)

Y_train = train_data["label"]
Y_test = test_data["label"]

print("nn: LSTM - Long Short-Term Memory")
model = Sequential()
model.add(Embedding(input_dim=TOKENIZER_MAX_WORDS, output_dim=128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

print(model.summary())

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("training the model")
model.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)

def save_lstm_model(model, epochs, batch_size, review_max_length):
  model_name = f"lstm_{epochs}_epochs_{batch_size}_batch_size_{review_max_length}_review_max_length"
  model.save(f"./models/{model_name}.keras")
  model.save_weights(f"./models/{model_name}.weights.h5")

save_lstm_model(model, NUM_EPOCHS, BATCH_SIZE, REVIEW_MAX_LENGTH)

loss, accuracy = model.evaluate(X_test, Y_test)
print(f"test loss:     {loss}")
print(f"test lccuracy: {accuracy}")