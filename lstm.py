import pandas as pd
from tqdm import tqdm
import tensorflow as tf
tf.random.set_seed(1337)

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2

TOKENIZER_MAX_WORDS = 10000

REVIEW_LENGTH = 250
BATCH_SIZE = 64

data = pd.concat([chunk for chunk in tqdm(pd.read_csv("./clean_dataset.csv", chunksize=1000), desc="loading the dataset")])

train_data, test_data = train_test_split(data, test_size=TEST_SPLIT, random_state=1337, stratify=data["label"])

print(f"train data shape: {train_data.shape}")
print(f"test data shape:  {test_data.shape}")

print(f"tokenizing the test dataset")
tokenizer = Tokenizer(num_words=TOKENIZER_MAX_WORDS)
tokenizer.fit_on_texts(train_data["text"])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["text"]), maxlen=REVIEW_LENGTH)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["text"]), maxlen=REVIEW_LENGTH)

Y_train = train_data["label"]
Y_test = test_data["label"]

print("building the LSTM")
# ------------------ NEW CODE --------------------
from tensorflow.keras.callbacks import ModelCheckpoint

EMBED_DIM = 64 # 32
LSTM_OUT = 128 # 64

total_words = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(total_words, EMBED_DIM, input_length=REVIEW_LENGTH))
model.add(LSTM(LSTM_OUT, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print(model.summary())

checkpoint = ModelCheckpoint(
  'models/lstm.keras',
  monitor='accuracy',
  save_best_only=True,
  verbose=1
)

model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = 5, callbacks=[checkpoint])

metrics = model.evaluate(X_test, Y_test)