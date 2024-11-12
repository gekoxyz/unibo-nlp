import pandas as pd
import numpy as np

import torch

import nltk
from nltk.corpus import stopwords 
from collections import Counter
from sklearn.model_selection import train_test_split


train_path = 'imdb/plain_text/train-00000-of-00001.parquet'
test_path = 'imdb/plain_text/test-00000-of-00001.parquet'

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# print("**************************")
# print(df_train.shape)
# print(df_train.head()) 
# print("**************************")
# print(df_test.shape)
# print(df_test.head()) 

df = pd.concat([df_train, df_test], ignore_index=True, sort=False)
print("**************************")
print(df.shape)
print(df.head()) 

from string import punctuation
df['text'] = df['text'].apply(lambda x:''.join([c for c in x if c not in punctuation]))

# def remove_tags(string):
#   removelist = ""
#   result = re.sub('','',string)
#   result = re.sub('https://.*','',result)
#   result = re.sub(r'[^w'+removelist+']', ' ',result)
#   result = result.lower()
#   return result

# df['text']=df['text'].apply(lambda cw : remove_tags(cw))

stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
  st = ""
  for w in w_tokenizer.tokenize(text):
    st = st + lemmatizer.lemmatize(w) + " "
  return st
df['text'] = df.text.apply(lemmatize_text)

print("**************************")
print(df.shape)
print(df.head()) 


# EXPERIMENTAL
all_text2 = df['text'].tolist()
from collections import Counter
all_text2 = ' '.join(all_text2)
# create a list of words
words = all_text2.split()
# Count all the words using Counter Method
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)
vocab_to_int = {w:i for i, (w,c) in enumerate(sorted_words)}
vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
reviews_split = df['text'].tolist()

reviews_int = []
for review in reviews_split:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)
print (reviews_int[0:3])

labels_split = df['label'].tolist()
encoded_labels = [1 if label =='positive' else 0 for label in labels_split]
encoded_labels = np.array(encoded_labels)

import pandas as pd
import matplotlib.pyplot as plt
reviews_len = [len(x) for x in reviews_int]
# pd.Series(reviews_len).hist()
# plt.show()
pd.Series(reviews_len).describe()

def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    
    for i, review in enumerate(reviews_int):
      review_len = len(review)
        
      if review_len <= seq_length:
        zeroes = list(np.zeros(seq_length-review_len))
        new = zeroes+review
      elif review_len > seq_length:
        new = review[0:seq_length]
        
      features[i,:] = np.array(new)
    
    return features

features = pad_features(reviews_int,200)
print (features[:10,:])
len_feat = len(features)
split_frac = 0.8
train_x = features[0:int(split_frac*len_feat)]
train_y = encoded_labels[0:int(split_frac*len_feat)]
remaining_x = features[int(split_frac*len_feat):]
remaining_y = encoded_labels[int(split_frac*len_feat):]
valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
valid_y = remaining_y[0:int(len(remaining_y)*0.5)]
test_x = remaining_x[int(len(remaining_x)*0.5):]
test_y = remaining_y[int(len(remaining_y)*0.5):]

train_y = np.array(train_y)
test_y = np.array(test_y)
valid_y = np.array(valid_y)

import torch
from torch.utils.data import DataLoader, TensorDataset
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
# dataloaders
batch_size = 50
# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)
print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)
