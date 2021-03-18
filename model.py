import torch
import torch.nn as nn


# Initializing all parameters of the model
def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


def categorical_accuracy(onehot, preds, y, tag_pad_idx):
    if onehot == True:
        # get the index of the max probability from onehot format
        preds = preds.argmax(dim=1, keepdim=True)
        non_pad_elements = (y != tag_pad_idx).nonzero()
        correct = preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    else:
        non_pad_elements = (y != tag_pad_idx).nonzero()
        correct = preds[non_pad_elements].eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, pad_idx, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(self.dropout(outputs))

        return predictions


def train_bilstm(model, iterator, optimizer, criterion, tag_pad_idx, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        word = batch['word_tensor']
        tag = batch['tag_tensor']
        word = word.to(device)
        tag = tag.to(device)
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


def evaluate_bilstm(model, iterator, criterion, tag_pad_idx, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    # Do not update any parameters
    with torch.no_grad():
        for batch in iterator:
            word = batch['word_tensor']
            tag = batch['tag_tensor']
            word = word.to(device)
            tag = tag.to(device)
            predictions = model(word)
            predictions = predictions.view(-1, predictions.shape[-1])
            tag = tag.view(-1)
            loss = criterion(predictions, tag)
            acc = categorical_accuracy(True, predictions, tag, tag_pad_idx)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# BiLSTM_crf
def argmax(vec):
    # max(data，dim)
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the batch forward algorithm
def log_sum_exp_bacth(vec):
    max_score_vec = torch.max(vec, dim=1)[0]
    max_score_broadcast = max_score_vec.view(vec.shape[0], -1).expand(vec.shape[0], vec.size()[1])
    # max_score_boradcast = [seq_length, output_dim]
    return max_score_vec + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))


class BiLSTM_CRF(nn.Module):
    def __init__(self, input_dim, tag2idx, embedding_dim, hidden_dim, n_layers, pad_idx, dropout):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.tag2idx = tag2idx
        self.output_dim = len(tag2idx)
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, self.output_dim)
        # transition matrix, entry i,j is the score of from j to i
        # transition = [output_dim, output_dim]
        self.transitions = nn.Parameter(torch.randn(self.output_dim, self.output_dim))
        self.transitions.data[self.tag2idx['START_TAG'], :] = -10000
        self.transitions.data[:, self.tag2idx['STOP_TAG']] = -10000

    # get emit_score via lstm
    def _get_lstm_features(self, sentence):
        embedded = self.dropout(self.embedding(sentence))
        # embedded  = [batch_size, seq_length, embedding_dim * 2]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out = [batch_size, seq_length, hidden_dim]
        lstm_feats = self.fc(self.dropout(lstm_out))
        # lstm_feats = [seq_length, output_dim]
        return lstm_feats

    def _forward_alg_parallel(self, batch_feats):
        # batch_feats = [seq_length, batch_size, output_dim]
        init_alphas = torch.full((batch_feats.shape[0], self.output_dim), -10000.)
        init_alphas[:, self.tag2idx['START_TAG']] = 0.
        # init_alphas = [seq_length, output_dim]
        forward_var = init_alphas

        convert_feats = batch_feats.permute(1, 0, 2)
        # convert_feats = [batch_size, seq_length, output_dim]
        for feat in convert_feats:
            # feat = [seq_length, ouput_dim]
            alphas_t = []
            for next_tag in range(self.output_dim):
                # tag = 0,1,2,3,....,19
                emit_score = feat[:, next_tag].view(batch_feats.shape[0], -1).expand(batch_feats.shape[0],
                                                                                     self.output_dim)
                # emit_score = [seq_length, output_dim]
                trans_score = self.transitions[next_tag].view(1, -1).repeat(batch_feats.shape[0], 1)
                # trans_score = [seq_length, output_dim]
                next_tag_var = forward_var + trans_score + emit_score
                # next_tag_var = [seq_length, output_dim]
                alphas_t.append(log_sum_exp_bacth(next_tag_var))
            forward_var = torch.stack(alphas_t).permute(1, 0)

        terminal_var = forward_var + self.transitions[self.tag2idx['STOP_TAG']].view(1, -1).repeat(batch_feats.shape[0],
                                                                                                   1)
        alpha = log_sum_exp_bacth(terminal_var)
        return alpha

    # find the best path
    def _viterbi_decode(self, batch_feats):
        best_paths = []
        for feats in batch_feats:
            backpointers = []

            init_vvars = torch.full((1, self.output_dim), -10000.)
            init_vvars[0][self.tag2idx['START_TAG']] = 0.

            forward_var = init_vvars

            for feat in feats:
                bptrs_t = []
                viterbivars_t = []
                for next_tag in range(self.output_dim):
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            terminal_var = forward_var + self.transitions[self.tag2idx['STOP_TAG']]
            best_tag_id = argmax(terminal_var)
            # Solve the optimal path from bcakpointers
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            start = best_path.pop()
            # double check
            assert start == self.tag2idx['START_TAG']
            # Return to normal sequence
            best_path.reverse()
            best_paths.append(best_path)
        return best_paths

    def _score_sentence(self, batch_feats, batch_tags):
        # Gives the score of a provided tag sequence
        totalsocre_list = []
        for feats, tags in zip(batch_feats, batch_tags):
            score = torch.zeros(1)
            start_tag = torch.tensor([self.tag2idx['START_TAG']], dtype=torch.long)
            tags = torch.cat([start_tag, tags])
            for i, feat in enumerate(feats):
                score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            score = score + self.transitions[self.tag2idx['STOP_TAG'], tags[-1]]
            totalsocre_list.append(score)

        return torch.cat(totalsocre_list)

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg_parallel(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        tag_seq = self._viterbi_decode(lstm_feats)

        return torch.Tensor(tag_seq)


def train_bilstmcrf(model, iterator, optimizer, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        word = batch['word_tensor']
        tag = batch['tag_tensor']
        optimizer.zero_grad()
        predictions = model(word)
        loss = model.neg_log_likelihood(word, tag)
        acc = categorical_accuracy(False, predictions.view(-1), tag.view(-1), tag_pad_idx)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_bilstmcrf(model, iterator, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            word = batch['word_tensor']
            tag = batch['tag_tensor']
            predictions = model(word)
            loss = model.neg_log_likelihood(word, tag)
            acc = categorical_accuracy(False, predictions.view(-1), tag.view(-1), tag_pad_idx)
            epoch_loss += loss.item()  # / seq_length
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# crf
class CRFTagger():
    def __init__(self, train, test, pair):
        self.train = train
        self.test = test
        self.pair = pair

    def _word2crffeat(self, sent, i, pair):
        if pair:
            word = sent[i][0]
            features = {
                'word': word,
                'is_first': i == 0,
                'is_last': i == len(sent) - 1,
                'prev_word': '' if i == 0 else sent[i - 1][0],
                'next_word': '' if i == len(sent) - 1 else sent[i + 1][0],
                'has_hyphen': '-' in word,
                'is_numeric': word.isdigit(),
            }
        else:
            word = sent[i]
            features = {
                'bias': 1.0,
                'word': word,
                'is_first': i == 0,
                'is_last': i == len(sent) - 1,
                'prev_word': '' if i == 0 else sent[i - 1],
                'next_word': '' if i == len(sent) - 1 else sent[i + 1],
                'has_hyphen': '-' in word,
                'is_numeric': word.isdigit(),
            }
        return features

    def _sent2crffeat(self, sent):
        return [self._word2crffeat(sent, i, self.pair) for i in range(len(sent))]

    def _sent2crflabel(self, sent):
        if self.pair:
            return [label for i, label in sent]
        else:
            return [label for label in sent]

    def get_feature(self):
        x_train = [self._sent2crffeat(s) for s in self.train]
        y_train = [self._sent2crflabel(s) for s in self.train]
        x_test = [self._sent2crffeat(s) for s in self.test]
        y_test = [self._sent2crflabel(s) for s in self.test]

        return x_train, y_train, x_test, y_test


# random foreset
class RFTagger():
    def __init__(self, train, test, pair, PAD_IDX, word2idx, tag2idx):
        self.train = train
        self.test = test
        self.pair = pair
        self.PAD_IDX = PAD_IDX
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def _word2rffeat(self, sent, i, word2idx):
        if self.pair:
            word = word2idx[sent[i][0]]
            if len(sent) >= 5:
                if i == 0:
                    feature = [self.PAD_IDX, self.PAD_IDX, self.PAD_IDX, word, word2idx[sent[i + 1][0]]]
                elif i == 1:
                    feature = [self.PAD_IDX, self.PAD_IDX, word2idx[sent[i - 1][0]], word, word2idx[sent[i + 1][0]]]
                elif i == 2:
                    feature = [self.PAD_IDX, word2idx[sent[i - 2][0]], word2idx[sent[i - 1][0]], word,
                               word2idx[sent[i + 1][0]]]
                elif i == len(sent) - 1:
                    feature = [word2idx[sent[i - 3][0]], word2idx[sent[i - 2][0]], word2idx[sent[i - 1][0]], word,
                               self.PAD_IDX]
                else:
                    feature = [word2idx[sent[i - 3][0]], word2idx[sent[i - 2][0]], word2idx[sent[i - 1][0]], word,
                               word2idx[sent[i + 1][0]]]
            else:
                feature = [self.PAD_IDX, self.PAD_IDX, self.PAD_IDX, word, self.PAD_IDX]
        else:
            word = word2idx[sent[i]]
            if len(sent) >= 5:
                if i == 0:
                    feature = [self.PAD_IDX, self.PAD_IDX, self.PAD_IDX, word, word2idx[sent[i + 1]]]
                elif i == 1:
                    feature = [self.PAD_IDX, self.PAD_IDX, word2idx[sent[i - 1]], word, word2idx[sent[i + 1]]]
                elif i == 2:
                    feature = [self.PAD_IDX, word2idx[sent[i - 2]], word2idx[sent[i - 1]], word, word2idx[sent[i + 1]]]
                elif i == len(sent) - 1:
                    feature = [word2idx[sent[i - 3]], word2idx[sent[i - 2]], word2idx[sent[i - 1]], word, self.PAD_IDX]
                else:
                    feature = [word2idx[sent[i - 3]], word2idx[sent[i - 2]], word2idx[sent[i - 1]], word,
                               word2idx[sent[i + 1]]]
            else:
                feature = [self.PAD_IDX, self.PAD_IDX, self.PAD_IDX, word, self.PAD_IDX]
        return feature

    def _sent2rffeat(self, sent):
        return [self._word2rffeat(sent, i, self.word2idx) for i in range(len(sent))]

    def _sent2rflabel(self, sent):
        if self.pair:
            return [self.tag2idx[label] for i, label in sent]
        else:
            return [self.tag2idx[label] for label in sent]

    def _reload(self, sents):
        output = []
        for sent in sents:
            for item in sent:
                output.append(item)
        return output

    def get_feature(self):
        x_train = [self._sent2rffeat(s) for s in self.train]
        y_train = [self._sent2rflabel(s) for s in self.train]
        x_test = [self._sent2rffeat(s) for s in self.test]
        y_test = [self._sent2rflabel(s) for s in self.test]

        train_x = self._reload(x_train)
        test_x = self._reload(x_test)
        train_y = self._reload(y_train)
        test_y = self._reload(y_test)

        return train_x, test_x, train_y, test_y
