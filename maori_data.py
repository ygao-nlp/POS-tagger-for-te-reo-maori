import torch
from torch.utils.data import Dataset

import numpy as np
import re

import random

SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def transformer(sent):
    processed = []
    sent = re.sub('[" "]+', " ", sent)
    for item in sent.strip().split(' '):
        item = re.sub('\n', '', item)
        (word, tag) = item.split('|')
        if word.isdigit():
            tag = 'num'
        if word == '.' or word == '!' or word == '?':
            tag = 'punct'
        if tag == 'manner':
            tag = 'v'
        processed.append((word, tag))
    return processed

def helper(output):
    words, tags = [], []
    for sen in output:
        words.append(list(zip(*sen))[0])
        tags.append(list(zip(*sen))[1])

    word_dc, tag_dc = {}, {}
    for s in tags:
        for t in s:
            if t not in tag_dc:
                tag_dc[t] = 0
            tag_dc[t] += 1
    for s in words:
        for t in s:
            if t not in word_dc:
                word_dc[t] = 0
            word_dc[t] += 1

    idx2tag = [key for key in tag_dc]
    idx2tag.insert(0, '<pad>')
    tag2idx = {word: index for index, word in enumerate(idx2tag)}

    idx2word = [key for key in word_dc]
    idx2word.insert(0, '<pad>')
    idx2word.insert(1, '<unk>')
    word2idx = {word: index for index, word in enumerate(idx2word)}

    return words, tags, word_dc, tag_dc, idx2tag, tag2idx, idx2word, word2idx

def extend_TEM():
    path_to_file = 'maori-tag-string.txt'
    sentences, sent, output = [], [], []

    with open(path_to_file, encoding='utf-8') as f:
        sentences = f.readlines()

    for s in sentences:
        if len(s) > 5:
            sent.append(s)

    sent[442] = 'i|prep tae|v mai|dir hoki|n etahi|n o|prep ngā|num kau-matua|n kai-tohutohu|n kaunihera|n ;|punct ara|n ;|punct a|det te|det keea|n te|det rangiuawhe|n (|punct te|det arawa|n )|punct ;|punct tamahau|n mahuuku|n (|punct rongokako|n )|punct ;|punct hori|n ngatai|n (|punct tauranga|n )|punct ;|punct me|n waata|n te|det rangikotua|n (|punct matatua|n )|punct .|.'
    sent[445] = 'i|prep hui|n i|prep te|det mane|n a|det tae|v noa|conj ki|prep te|det hatarei|n te|det 25|n o|prep aerira|n ;|punct ka|tam koi|n ngā|num mahi|n a|det te|det huihuinga|n .|.'
    sent[609] = 'mo|n runga|n mo|n te|det whenua|n e|tam karangatia|n ana|tam e|tam mohiotia|n ana|tam ranei|n i|prep mua|n ko|prep arekohe|n te|det ingoa|n ;|punct ara|n ko|prep tetahi|n wahi|n o|prep te|det whenua|n e|tam whakaaturia|n ake|dir ana|tam i|prep roto|n i|prep te|det kuu|n aiti|n tuatahi|n ki|prep taua|det ture|n ;|punct a|det mo|n runga|n hoki|n mo|n te|det whakawākanga|n o|prep ngā|num take|n aanga|n ki|prep taua|det whenua|n .|.'
    sent[976] = 'natanahira|n te|det hurua|n ;|punct waka|n te|det uhi|n ;|punct tinioaka|n ;|punct me|n etahi|n atu|dir ongohi|n ;|punct e|tam tata|n ana|tam ki|prep moehau|n kei|n te|det mai|dir ngā|num rohe|n'
    sent[1009] = 'tari|v o|prep te|det kooti|n whenua|n māori|n ;|punct akarana|n ;|punct oketoa|n 16|n ;|punct 1877|n .|.'

    for s in sent:
        output.append(transformer(s))

    random.shuffle(output)

    words, tags, word_dc, tag_dc, idx2tag, tag2idx, idx2word, word2idx = helper(output)

    return output, idx2tag, tag2idx, idx2word, word2idx, words, tags, word_dc, tag_dc

def noisy_TEM():
    path_to_file = 'maori-tag-string-noisy.txt'
    sentences, sent, output = [], [], []

    with open(path_to_file, encoding='utf-8') as f:
        sentences = f.readlines()

    for s in sentences:
        if len(s) > 5:
            sent.append(s)

    sent[442] = 'i|prep tae|v mai|dir hoki|n etahi|n o|prep ngā|num kau-matua|n kai-tohutohu|n kaunihera|n ;|punct ara|n ;|punct a|det te|det keea|n te|det rangiuawhe|n (|punct te|det arawa|n )|punct ;|punct tamahau|n mahuuku|n (|punct rongokako|n )|punct ;|punct hori|n ngatai|n (|punct tauranga|n )|punct ;|punct me|n waata|n te|det rangikotua|n (|punct matatua|n )|punct .|.'
    sent[445] = 'i|prep hui|n i|prep te|det mane|n a|det tae|v noa|conj ki|prep te|det hatarei|n te|det 25|n o|prep aerira|n ;|punct ka|tam koi|n ngā|num mahi|n a|det te|det huihuinga|n .|.'
    sent[609] = 'mo|n runga|n mo|n te|det whenua|n e|tam karangatia|n ana|tam e|tam mohiotia|n ana|tam ranei|n i|prep mua|n ko|prep arekohe|n te|det ingoa|n ;|punct ara|n ko|prep tetahi|n wahi|n o|prep te|det whenua|n e|tam whakaaturia|n ake|dir ana|tam i|prep roto|n i|prep te|det kuu|n aiti|n tuatahi|n ki|prep taua|det ture|n ;|punct a|det mo|n runga|n hoki|n mo|n te|det whakawākanga|n o|prep ngā|num take|n aanga|n ki|prep taua|det whenua|n .|.'
    sent[976] = 'natanahira|n te|det hurua|n ;|punct waka|n te|det uhi|n ;|punct tinioaka|n ;|punct me|n etahi|n atu|dir ongohi|n ;|punct e|tam tata|n ana|tam ki|prep moehau|n kei|n te|det mai|dir ngā|num rohe|n'
    sent[1009] = 'tari|v o|prep te|det kooti|n whenua|n māori|n ;|punct akarana|n ;|punct oketoa|n 16|n ;|punct 1877|n .|.'

    for s in sent:
        output.append(transformer(s))

    random.shuffle(output)

    words, tags, word_dc, tag_dc, idx2tag, tag2idx, idx2word, word2idx = helper(output)

    return output, words, tags

def TEM():
    path_to_file = 'maori-tag-string(part).txt'
    sentences, sent, output = [], [], []
    with open(path_to_file, encoding='utf-8') as f:
        sentences = f.readlines()

    for s in sentences:
        if len(s) > 5:
            sent.append(s)

    for s in sent:
        output.append(transformer(s))

    random.shuffle(output)

    words, tags, word_dc, tag_dc, idx2tag, tag2idx, idx2word, word2idx = helper(output)

    return output, idx2tag, tag2idx, idx2word, word2idx, words, tags, word_dc, tag_dc

class MaoriDataset(Dataset):
    def __init__(self, word_seq, tag_seq, word2idx, tag2idx, word_dc, seq_length):
        self.word_seq = word_seq
        self.tag_seq = tag_seq
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.word_dc = word_dc
        self.seq_length = seq_length
        self.unk_word = set()
        self.unk_tag = set()

    def __len__(self):
        return len(self.word_seq)

    def __getitem__(self, idx):
        # init torch tensors, note that 0 is the padding index
        word_tensor = torch.zeros(self.seq_length, dtype=torch.long)
        tag_tensor = torch.zeros(self.seq_length, dtype=torch.long)
        # Get sentence pair
        word_sentence = self.word_seq[idx]
        tag_sentence = self.tag_seq[idx]
        # Load word indices
        for i, word in enumerate(word_sentence):
            if word in self.word2idx and self.word_dc[word] > 2:
                word_tensor[i] = self.word2idx[word]
            else:
                word_tensor[i] = self.word2idx['<unk>']
                self.unk_word.add(word)
                # Load tag indices
        for i, tag in enumerate(tag_sentence):
            tag_tensor[i] = self.tag2idx[tag]

        return {'word_tensor': word_tensor, 'tag_tensor': tag_tensor, }