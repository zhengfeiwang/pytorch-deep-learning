import os
import nltk
from nltk.tokenize import RegexpTokenizer
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sent_tokenize = nltk.sent_tokenize
tokenize = RegexpTokenizer(r'\w+').tokenize


def read_data(dir):
    texts = []
    labels = []
    for label_type in ['neg', 'pos']:
        type_dir = os.path.join(dir, label_type)
        for filename in os.listdir(type_dir):
            if filename[-4:] == '.txt':
                text_file = open(os.path.join(type_dir, filename))
                texts.append(text_file.read())
                text_file.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
    return texts, labels

# Function for tokenizing the text
def flatten(l):
    return [item for sublist in l for item in sublist]


def get_words(text):
    return (flatten([tokenize(sentence) for sentence in sent_tokenize(text)]))


def vocab_indices(word2vec, text, sequence_length):
    return [word2vec.stoi[word] for word in get_words(text)[0 : sequence_length]]


def get_word_vectors(word2vec, text, sequence_length):
    indices = vocab_indices(word2vec, text, sequence_length)
    word_vectors = torch.cat([word2vec.vectors[i].view(-1, 1) for i in indices]).view(len(indices), 1, -1)
    return word_vectors.view(1, len(word_vectors), -1)


def get_batch(data, start_idx, end_idx, word2vec, sequence_length):
    batch = data[start_idx : end_idx]
    texts, lables = zip(*batch)
    
    inputs = [get_word_vectors(word2vec, text, sequence_length) for text in texts]
    labels = torch.LongTensor(lables)

    seq_lens = torch.LongTensor([vector.shape[1] for vector in inputs])
    embedding_dim = inputs[0].shape[2]
    
    batch_inputs = torch.zeros((len(seq_lens), seq_lens.max(), embedding_dim))
    for idx, (seq, seq_len) in enumerate(zip(inputs, seq_lens)):
        batch_inputs[idx, :seq_len] = seq

    seq_lens, perm_idx = seq_lens.sort(0, descending=True)
    batch_inputs = batch_inputs[perm_idx]
    batch_inputs = pack_padded_sequence(batch_inputs, seq_lens.numpy(), batch_first=True)
    labels = labels[perm_idx]
    return (batch_inputs, labels)
