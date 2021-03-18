import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

import numpy as np
import sklearn_crfsuite
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from Maori.model import CRFTagger, RFTagger
from Maori.model import BiLSTM
from Maori.model import BiLSTM_CRF, categorical_accuracy

import time
import random

# set random seed
SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

print(torch.__version__)

if torch.cuda.is_available():
  device = torch.device('cuda')
  torch.cuda.set_device(1)
else:
  device = torch.device('cpu')

"""# preparing data"""
TEXT = data.Field(lower = True)
UD_TAGS = data.Field(unk_token = None)
fields = (("text", TEXT), ("udtags", UD_TAGS), (None,None))

train_data, dev_data, test_data = datasets.UDPOS.splits(fields)

# using train_data create dictionary
MIN_FREQ = 2
TEXT.build_vocab(train_data,min_freq=MIN_FREQ,vectors="glove.6B.100d",unk_init=torch.Tensor.normal_)
UD_TAGS.build_vocab(train_data)

BATCH_SIZE = 128
batch_train, batch_dev, batch_test = data.BucketIterator.splits((train_data, dev_data, test_data), batch_size = BATCH_SIZE, device = device)

PAD_IDX = 1
pair = False

"""# Building CRF"""
# model from sklearn
crffeat = CRFTagger(train_data+dev_data, test_data, pair)
X_train=[crffeat._sent2crffeat(s.text) for s in train_data+dev_data]
y_train=[crffeat._sent2crflabel(s.udtags) for s in train_data+dev_data]
X_test=[crffeat._sent2crffeat(s.text) for s in test_data]
y_test=[crffeat._sent2crflabel(s.udtags) for s in test_data]

crf = sklearn_crfsuite.CRF(max_iterations=50)
crf.fit(X_train, y_train)

crf_pred = crf.predict(X_test)
labels=list(crf.classes_)

test_y, crf_y = [], []
for i in y_test:
  test_y.extend(i)
for i in crf_pred:
  crf_y.extend(i)

crf_acc = accuracy_score(test_y, crf_y)*100
print(f"CRF testing accuracy: {crf_acc:.2f}%")

"""# Building Random Forest"""
# model from sklearn
rffeat = RFTagger(train_data+dev_data, test_data, pair, PAD_IDX, TEXT.vocab.stoi, UD_TAGS.vocab.stoi)
X_train=[rffeat._sent2rffeat(s.text) for s in train_data+dev_data]
y_train=[rffeat._sent2rflabel(s.udtags) for s in train_data+dev_data]
X_test=[rffeat._sent2rffeat(s.text) for s in test_data]
y_test=[rffeat._sent2rflabel(s.udtags) for s in test_data]

def reload(sents):
  output = []
  for sent in sents:
    for item in sent:
      output.append(item)
  return output

train_x = reload(X_train)
test_x = reload(X_test)
train_y = reload(y_train)
test_y = reload(y_test)

rf = RandomForestClassifier(n_estimators = 100, random_state = SEED)

rf.fit(train_x, train_y)
rf_y = rf.predict(test_x)

rf_acc = accuracy_score(test_y, rf_y)*100
print(f"RF testing accuracy: {rf_acc:.2f}%")

# setting parameters
from Maori.model import init_weights, epoch_time

START_TAG = 'START_TAG'
STOP_TAG = 'STOP_TAG'
INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(UD_TAGS.vocab.stoi)
TAG2IDX = UD_TAGS.vocab.stoi
EMBEDDING_DIM = 100
HIDDEN_DIM = 32
N_LAYERS = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
DROPOUT = 0.3

"""# Building Bi-LSTM Model"""
print('----------Bi-LSTM----------')

bilstm = BiLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, PAD_IDX, DROPOUT)
bilstm.apply(init_weights)

optimizer = optim.Adam(bilstm.parameters(),lr=0.003)
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

bilstm = bilstm.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion, tag_pad_idx):
  epoch_loss = 0
  epoch_acc = 0
  model.train()
  for batch in iterator:
    word = batch.text
    tag = batch.udtags
    optimizer.zero_grad()
    predictions = model(word)
    predictions = predictions.view(-1, predictions.shape[-1])
    tag = tag.view(-1)
    loss = criterion(predictions, tag)
    acc = categorical_accuracy(True, predictions, tag, tag_pad_idx)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):
  epoch_loss = 0
  epoch_acc = 0
  # Do not update any parameters
  model.eval()
  with torch.no_grad():
    for batch in iterator:
      word = batch.text
      tag = batch.udtags
      predictions = model(word)
      predictions = predictions.view(-1, predictions.shape[-1])
      tag = tag.view(-1)
      loss = criterion(predictions, tag)
      acc = categorical_accuracy(True, predictions, tag, tag_pad_idx)
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 50
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
  start_time = time.time()  
  train_loss, train_acc = train(bilstm, batch_train, optimizer, criterion, PAD_IDX)
  valid_loss, valid_acc = evaluate(bilstm, batch_dev, criterion, PAD_IDX)
  end_time = time.time()
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)

  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

bilstm_test_loss, bilstm_test_acc = evaluate(bilstm, batch_test, criterion, PAD_IDX)

print(f'Test Loss: {bilstm_test_acc:.3f} |  Test Acc: {bilstm_test_acc*100:.2f}%')

"""# Building Bi-LSTM + CRF"""
# model based on pytorch bilstm+crf
print('----------Bi-LSTM + CRF----------')
UD_TAGS.vocab.stoi['START_TAG'] = 18
UD_TAGS.vocab.stoi['STOP_TAG'] = 19

bilstm_crf = BiLSTM_CRF(INPUT_DIM, TAG2IDX, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, PAD_IDX, DROPOUT)
bilstm_crf.apply(init_weights)

optimizer = optim.Adam(bilstm_crf.parameters(),lr=0.003)

def train(model, iterator, optimizer, tag_pad_idx):
  epoch_loss = 0
  epoch_acc = 0
  model.train()
  for batch in iterator:
    word = batch.text
    tag = batch.udtags
    word = word.cpu()
    tag = tag.cpu()
    optimizer.zero_grad()
    predictions = model(word)
    loss = model.neg_log_likelihood(word, tag)        
    acc = categorical_accuracy(False, predictions.view(-1), tag.view(-1), tag_pad_idx)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    epoch_acc += acc.item()
      
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, tag_pad_idx):
  epoch_loss = 0
  epoch_acc = 0
  # Do not update any parameters
  model.eval()
  with torch.no_grad():
    for batch in iterator:
      word = batch.text
      tag = batch.udtags
      word = word.cpu()
      tag = tag.cpu()
      predictions = model(word)
      loss = model.neg_log_likelihood(word, tag)   
      acc = categorical_accuracy(False, predictions.view(-1), tag.view(-1), tag_pad_idx)
      epoch_loss += loss.item()
      epoch_acc += acc.item()
      
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 50
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
  start_time = time.time()  
  train_loss, train_acc = train(bilstm_crf, batch_train, optimizer, PAD_IDX)
  valid_loss, valid_acc = evaluate(bilstm_crf, batch_dev, PAD_IDX)

  end_time = time.time()
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)

  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

bc_test_loss, bc_test_acc = evaluate(bilstm_crf, batch_test, PAD_IDX)

print(f'Test Loss: {bc_test_loss:.3f} |  Test Acc: {bc_test_acc*100:.2f}%')