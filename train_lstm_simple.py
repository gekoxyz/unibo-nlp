import itertools
import pandas as pd
import tensorflow as tf
tf.random.set_seed(1337)

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2

TOKENIZER_MAX_WORDS = 10000
EPOCHS = 5
EMBED_DIM = 64
LSTM_OUT = 128

early_stopping = EarlyStopping(
  monitor="val_loss",
  patience=2,  # wait 2 epochs after the best epoch
  restore_best_weights=True
)

hyperparameters = {
  "BATCH_SIZE": [64, 128],
  "REVIEW_MAX_LENGTH": [250, 500]
}

def save_lstm_model(model, batch_size, review_max_length):
  model_name = f"lstm_{batch_size}_batch_size_{review_max_length}_review_max_length"
  model.save(f"./models/{model_name}.keras")

data = pd.read_csv("./clean_dataset.csv")

train_data, test_data = train_test_split(data, test_size=TEST_SPLIT, random_state=1337, stratify=data["label"])
print(f"train data shape: {train_data.shape}")
print(f"test data shape:  {test_data.shape}")

print(f"tokenizing the test dataset")
tokenizer = Tokenizer(num_words=TOKENIZER_MAX_WORDS)
tokenizer.fit_on_texts(train_data["text"])


for hyperparameters_combinations in itertools.product(*hyperparameters.values()):
  current_hyperparameters = dict(zip(hyperparameters.keys(), hyperparameters_combinations))
  print(f"training with: {current_hyperparameters}")

  X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["text"]), maxlen=current_hyperparameters["REVIEW_MAX_LENGTH"])
  X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["text"]), maxlen=current_hyperparameters["REVIEW_MAX_LENGTH"])

  Y_train = train_data["label"]
  Y_test = test_data["label"]

  print(f"creating the model")
  model = Sequential()
  model.add(Embedding(input_dim=TOKENIZER_MAX_WORDS, output_dim=EMBED_DIM))
  model.add(LSTM(LSTM_OUT, dropout=0.2, recurrent_dropout=0.2))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
  print(model.summary())

  model_name = f"lstm_{current_hyperparameters['BATCH_SIZE']}_batch_size_{current_hyperparameters['REVIEW_MAX_LENGTH']}_review_max_length"

  model_checkpoint = ModelCheckpoint(
    f"models/{model_name}.keras",
    monitor='accuracy',
    save_best_only=True,
    verbose=1
  )

  print("training the model")
  history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=current_hyperparameters["BATCH_SIZE"], validation_split=VALIDATION_SPLIT, callbacks=[early_stopping, model_checkpoint])

  print("saving the model")
  save_lstm_model(model, current_hyperparameters["BATCH_SIZE"], current_hyperparameters["REVIEW_MAX_LENGTH"])
