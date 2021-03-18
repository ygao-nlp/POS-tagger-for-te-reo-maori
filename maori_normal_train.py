import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import sklearn_crfsuite
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from Maori.maori_data import extend_TEM, MaoriDataset, TEM, noisy_TEM

from Maori.model import CRFTagger, RFTagger
from Maori.model import BiLSTM, train_bilstm, evaluate_bilstm
from Maori.model import BiLSTM_CRF, train_bilstmcrf, evaluate_bilstmcrf

SEED = 123

print(torch.__version__)

if torch.cuda.is_available():
  device = torch.device('cuda')
  torch.cuda.set_device(0)
else:
  device = torch.device('cpu')

ex_output, ex_idx2tag, ex_tag2idx, ex_idx2word, ex_word2idx, ex_words, ex_tags, ex_word_dc, ex_tag_dc = extend_TEM()
output, idx2tag, tag2idx, idx2word, word2idx, words, tags, word_dc, tag_dc = TEM()
no_output, no_words, no_tags = noisy_TEM()

ex_seq_length = max(len(seq) for seq in ex_output)

"""# Create dataset"""
train_len = int(0.8 * len(no_output))
dev_len = int(0.2 * len(no_output))

# data for crf and rf
shallow_train = no_output[: train_len]
shallow_dev = no_output[train_len:train_len + dev_len]
shallow_test = output

# data for bilstm and bilstm+crf
deep_train_word, deep_train_tag = no_words[:train_len], no_tags[:train_len]
deep_dev_word, deep_dev_tag = no_words[train_len:], no_tags[train_len:]
deep_test_word, deep_test_tag = words, tags

train_dataset = MaoriDataset(deep_train_word,deep_train_tag,ex_word2idx,ex_tag2idx,ex_word_dc,ex_seq_length)
dev_dataset = MaoriDataset(deep_dev_word,deep_dev_tag,ex_word2idx,ex_tag2idx,ex_word_dc,ex_seq_length)
test_dataset = MaoriDataset(deep_test_word,deep_test_tag,ex_word2idx,ex_tag2idx,ex_word_dc,ex_seq_length)

batch_train = DataLoader(train_dataset, batch_size=8, num_workers=0)
batch_dev = DataLoader(dev_dataset, batch_size=8, num_workers=0)
batch_test = DataLoader(test_dataset, batch_size=8, num_workers=0)

PAD_IDX = 0
pair = True

"""# Building CRF"""
crffeat = CRFTagger(shallow_train+shallow_dev, shallow_test, pair)
X_train, y_train, X_test, y_test = crffeat.get_feature()

crf = sklearn_crfsuite.CRF(max_iterations=50)
crf.fit(X_train, y_train)
crf_pred = crf.predict(X_test)

test_y, crf_y = [], []
for i in y_test:
  test_y.extend(i)
for i in crf_pred:
  crf_y.extend(i)

crf_acc = accuracy_score(test_y, crf_y)*100
print(f"Crf testing accuracy: {crf_acc:.2f}%")

"""# Building Random Forest"""
rffeat = RFTagger(shallow_train+shallow_dev, shallow_test, pair, PAD_IDX, ex_word2idx, ex_tag2idx)
train_x, test_x, train_y, test_y = rffeat.get_feature()

rf = RandomForestClassifier(n_estimators = 100, random_state = SEED)
rf.fit(train_x, train_y)
rf_y = rf.predict(test_x)

rf_acc = accuracy_score(test_y, rf_y)*100
print(f"RF testing accuracy: {rf_acc:.2f}%")

# setting parameters
from Maori.model import init_weights, epoch_time

INPUT_DIM = len(ex_idx2word)
EMBEDDING_DIM = 100
HIDDEN_DIM = 32
OUTPUT_DIM = len(ex_idx2tag)
N_LAYERS = 1
PAD_IDX = 0
DROPOUT = 0.3

"""# Building Bi-LSTM Model"""
print('----------Bi-LSTM----------')

bilstm = BiLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, PAD_IDX, DROPOUT)
bilstm.apply(init_weights)

optimizer = optim.Adam(bilstm.parameters(),lr=0.003)
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

bilstm = bilstm.to(device)
criterion = criterion.to(device)

N_EPOCHS = 50
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
  start_time = time.time()  
  train_loss, train_acc = train_bilstm(bilstm, batch_train, optimizer, criterion, PAD_IDX, device)
  valid_loss, valid_acc = evaluate_bilstm(bilstm, batch_dev, criterion, PAD_IDX, device)
  end_time = time.time()
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      torch.save(bilstm.state_dict(), 'noisy_bilstm.pt')
  if (epoch+1) % 10 == 0:
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

bilstm.load_state_dict(torch.load('noisy_bilstm.pt'))

bilstm_test_loss, bilstm_test_acc = evaluate_bilstm(bilstm, batch_test, criterion, PAD_IDX, device)

print(f'Test Loss: {bilstm_test_acc:.3f} |  Test Acc: {bilstm_test_acc*100:.2f}%')

"""# Building Bi-LSTM + CRF"""
print('----------Bi-LSTM + CRF----------')

ex_tag2idx['START_TAG'] = 14
ex_tag2idx['STOP_TAG'] = 15
TAG2IDX = ex_tag2idx

bilstm_crf = BiLSTM_CRF(INPUT_DIM, TAG2IDX, EMBEDDING_DIM, HIDDEN_DIM,  N_LAYERS, PAD_IDX, DROPOUT)
bilstm_crf.apply(init_weights)

optimizer = optim.Adam(bilstm_crf.parameters(),lr=0.003)

N_EPOCHS = 50
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
  start_time = time.time()  
  train_loss, train_acc = train_bilstmcrf(bilstm_crf, batch_train, optimizer, PAD_IDX)
  valid_loss, valid_acc = evaluate_bilstmcrf(bilstm_crf, batch_dev, PAD_IDX)
  
  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(bilstm_crf.state_dict(), 'noisy_bilstm_crf.pt')
  if (epoch+1) % 10 == 0:
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

bc_test_loss, bc_test_acc = evaluate_bilstmcrf(bilstm_crf, batch_test, PAD_IDX)
print(f'Test Loss: {bc_test_loss:.3f} |  Test Acc: {bc_test_acc*100:.2f}%')

bilstm_crf.load_state_dict(torch.load('noisy_bilstm_crf.pt'))

bc_test_loss, bc_test_acc = evaluate_bilstmcrf(bilstm_crf, batch_test, PAD_IDX)
print(f'Test Loss: {bc_test_loss:.3f} |  Test Acc: {bc_test_acc*100:.2f}%')