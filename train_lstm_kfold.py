import pandas as pd
import tensorflow as tf
tf.random.set_seed(1337)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
K_FOLDS = 5

TOKENIZER_MAX_WORDS = 10000
EPOCHS = 5
EMBED_DIM = 64
LSTM_OUT = 128

early_stopping = EarlyStopping(
  monitor="val_loss",
  patience=2,  # wait 2 epochs after the best epoch
  restore_best_weights=True
)

tmp_hyp_combinations = [
  {"BATCH_SIZE": 64, "REVIEW_MAX_LENGTH": 250},
  {"BATCH_SIZE": 128, "REVIEW_MAX_LENGTH": 500}
]

def save_lstm_model(model, batch_size, review_max_length, fold_num):
  model_name = f"lstm_{batch_size}_batch_size_{review_max_length}_review_max_length_{fold_num}_fold"
  model.save(f"./models/{model_name}.keras")

data = pd.read_csv("./clean_dataset.csv")

skf = StratifiedKFold(n_splits=5, random_state=1337, shuffle=True)

for fold_num, (train_idx, val_idx) in enumerate(skf.split(data["text"], data["label"]), start=1):
  print(f"fold {fold_num}")
  print("================================================================")
  train_data = data.iloc[train_idx]
  test_data = data.iloc[val_idx]

  print("tokenizing the test dataset")
  tokenizer = Tokenizer(num_words=TOKENIZER_MAX_WORDS)
  tokenizer.fit_on_texts(train_data["text"])

  for hyperparameters in tmp_hyp_combinations:
    print(f"training with: {hyperparameters} on the fold {fold_num}")

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["text"]), maxlen=hyperparameters["REVIEW_MAX_LENGTH"])
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["text"]), maxlen=hyperparameters["REVIEW_MAX_LENGTH"])

    Y_train = train_data["label"]
    Y_test = test_data["label"]

    print(f"creating the model")
    model = Sequential()
    model.add(Embedding(input_dim=TOKENIZER_MAX_WORDS, output_dim=EMBED_DIM))
    model.add(LSTM(LSTM_OUT, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(model.summary())

    model_name = f"lstm_{hyperparameters['BATCH_SIZE']}_batch_size_{hyperparameters['REVIEW_MAX_LENGTH']}_review_max_length_{fold_num}_fold"

    model_checkpoint = ModelCheckpoint(
      f"models/{model_name}.keras",
      monitor='accuracy',
      save_best_only=True,
      verbose=1
    )

    print("training the model")
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=hyperparameters["BATCH_SIZE"], validation_split=VALIDATION_SPLIT, callbacks=[early_stopping, model_checkpoint])

    print("saving the model")
    save_lstm_model(model, hyperparameters["BATCH_SIZE"], hyperparameters["REVIEW_MAX_LENGTH"], fold_num)

    print("testing the model and generating the confusion matrix")
    y_pred = model.predict(X_test)

    # Convert predictions to class labels
    if y_pred.shape[-1] > 1:  # Multi-class classification
      y_pred_labels = np.argmax(y_pred, axis=1)
      y_true_labels = np.argmax(Y_test, axis=1)
    else:  # Binary classification
      y_pred_labels = (y_pred > 0.5).astype(int).squeeze()
      y_true_labels = Y_test.squeeze()

    print("calculating metrics")
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

    # Save metrics to a text file
    with open(f'{model_name}.txt', 'w') as f:
      f.write(f"Accuracy:  {accuracy:.4f}\n")
      f.write(f"Precision: {precision:.4f}\n")
      f.write(f"Recall:    {recall:.4f}\n")
      f.write(f"F1 Score:  {f1:.4f}\n")

    # Plot and save confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true_labels), 
                yticklabels=np.unique(y_true_labels))
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(f'{model_name}_confusion_matrix.png')  # Save the confusion matrix as an image
    plt.close()  # Close the plot to free up memory

  print("================================================================")