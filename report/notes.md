# Steps to train the model

## Clean the dataset

first attempt: no lemmatization, just removal of noise (links, html, punctuation) by using clean_dataset.ipynb

trained with sentiment_model_nn.ipynb

### hyperparameters and dataset split for imdb_processed_full.csv:
CORPUS_MAX_SIZE = 752

train_size = 0.8
validation_size = 0.5

batch_size = 128

vocab_size = len(word2int)
output_size = 1
embedding_size = 256
hidden_size = 512
n_layers = 2
dropout=0.25

results:
              precision    recall  f1-score   support

           0       0.83      0.83      0.83      5004
           1       0.83      0.83      0.83      4996

    accuracy                           0.83     10000
   macro avg       0.83      0.83      0.83     10000
weighted avg       0.83      0.83      0.83     10000

