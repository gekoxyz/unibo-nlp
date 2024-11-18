# Steps to train the model

## Random seeds
np.random.seed(1337)
torch seed 1337
gpu torch seed 1337

## Hyperparameters

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

lr = 0.001
criterion = nn.BCELoss()
optim = Adam(model.parameters(), lr=lr)
grad_clip = 5
epochs = 8
es_limit = 5

## Results

### first attempt: no lemmatization, just removal of noise (links, html, punctuation) by using clean_dataset.ipynb

SEED 1337
trained with sentiment_model_nn.ipynb
output filename: sentiment_lstm.pt
results:
              precision    recall  f1-score   support

           0       0.90      0.80      0.85      5620
           1       0.77      0.88      0.82      4380

    accuracy                           0.84     10000
   macro avg       0.84      0.84      0.83     10000
weighted avg       0.84      0.84      0.84     10000



### second attempt: lemmatization included and advanced cleaning

SEED 1337
trained with sentiment_model_nn.ipynb
output filename: lemmatized_model.pt
results:
              precision    recall  f1-score   support

           0       0.89      0.80      0.84      5538
           1       0.78      0.87      0.82      4462

    accuracy                           0.83     10000
   macro avg       0.83      0.83      0.83     10000
weighted avg       0.84      0.83      0.83     10000


### 3 hidden layers and CORPUS_MAX_SIZE 1024
SEED 1337
trained with sentiment_model_nn.ipynb
output filename: lemmatized_model_3hidden.pt
              precision    recall  f1-score   support

           0       0.86      0.85      0.86      5001
           1       0.85      0.86      0.86      4999

    accuracy                           0.86     10000
   macro avg       0.86      0.86      0.86     10000
weighted avg       0.86      0.86      0.86     10000

