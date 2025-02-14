import itertools
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
tf.random.set_seed(1337)

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2

TOKENIZER_MAX_WORDS = 10000

early_stopping = EarlyStopping(
  monitor="val_loss",
  patience=2,  # Wait 2 epochs after the best epoch
  restore_best_weights=True  # Revert to the best model
)

hyperparameters = {
  "NUM_EPOCHS": [3, 6],
  "BATCH_SIZE": [64, 128],
  "REVIEW_MAX_LENGTH": [500, 1000]
}

def save_lstm_model(model, epochs, batch_size, review_max_length):
  model_name = f"lstm_{epochs}_epochs_{batch_size}_batch_size_{review_max_length}_review_max_length"
  model.save(f"./models/{model_name}.keras")
  model.save_weights(f"./models/{model_name}.weights.h5")

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

  print("nn: LSTM - Long Short-Term Memory")
  model = Sequential()
  model.add(Embedding(input_dim=TOKENIZER_MAX_WORDS, output_dim=64))
  model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01)))
  model.add(Dense(1, activation="sigmoid"))

  print(model.summary())

  model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

  print("training the model")
  history = model.fit(X_train, Y_train, epochs=current_hyperparameters["NUM_EPOCHS"], batch_size=current_hyperparameters["BATCH_SIZE"], validation_split=VALIDATION_SPLIT, callbacks=[early_stopping])

  save_lstm_model(model, current_hyperparameters["NUM_EPOCHS"], current_hyperparameters["BATCH_SIZE"], current_hyperparameters["REVIEW_MAX_LENGTH"])

  loss, accuracy = model.evaluate(X_test, Y_test)
  print(f"LSTM with hyperparameters {current_hyperparameters}")
  print(f"test loss:     {loss}")
  print(f"test accuracy: {accuracy}")

  with open("hyp_lstm.txt", "a") as file:
    file.write(f"LSTM with hyperparameters {current_hyperparameters}\n")
    file.write(f"test loss:     {loss}\n")
    file.write(f"test accuracy: {accuracy}\n")

  # TODO: THIS DOESN'T HAVE ENOUGH DATA
  with open("training_metrics.txt", "a") as file:
    file.write(f"LSTM with hyperparameters {current_hyperparameters}\n")
    file.write(f"{history}\n")
    file.write(f"--------------------------------------------------------")
