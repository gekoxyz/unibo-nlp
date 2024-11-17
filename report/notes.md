# Steps to train the model

## Clean the dataset

first attempt: no lemmatization, just removal of noise (links, html, punctuation) by using clean_dataset.ipynb

trained with sentiment_model_nn.ipynb

hyperparameters and dataset split:


results:
              precision    recall  f1-score   support

           0       0.83      0.83      0.83      5004
           1       0.83      0.83      0.83      4996

    accuracy                           0.83     10000
   macro avg       0.83      0.83      0.83     10000
weighted avg       0.83      0.83      0.83     10000

