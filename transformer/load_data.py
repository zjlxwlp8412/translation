# encoding: utf-8

import tensorflow as tf
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from transformer import parameters as params

def create_vocabs(file_list, vocab_file):
    word_counts = {
        "<PAD>": 100000000000,
        "<UNK>": 10000000000,
        "<S>": 1000000000,
        "</S>": 100000000
    }
    for file in file_list:
        with tf.gfile.Open(file, mode="r") as reader:
            for line in reader:
                tokens = WordPunctTokenizer().tokenize(line.strip())
                for token in tokens:
                    if token in word_counts:
                        word_counts[token] += 1
                    else:
                        word_counts[token] = 1
    word_counts = sorted(word_counts.items(), key=lambda d: d[1], reverse=True)
    with tf.gfile.Open(vocab_file, mode="w") as writer:
        for word_ct in word_counts:
            writer.write("{}\t{}\n".format(word_ct[0], word_ct[1]))

def load_vocabs(vocab_file, min_count):
    word2idx = {}
    idx2word = {}
    with tf.gfile.Open(vocab_file, mode="r") as reader:
        idx = 0
        for line in reader:
            word_ct = line.strip().split("\t")
            if int(word_ct[1]) < min_count: break
            word2idx[word_ct[0]] = idx
            idx2word[idx] = word_ct[0]
            idx += 1
    return word2idx, idx2word

def create_inputs(file, word2idx, max_len, add_start=False):
    data = []
    with tf.gfile.Open(file, mode="r") as reader:
        for line in reader:
            line = line.strip()
            vec = np.zeros(shape=[max_len], dtype=np.int32)
            tokens = WordPunctTokenizer().tokenize(line)[:max_len]
            if add_start:
                tokens = ["<S>"] + tokens[1:]
            for i in range(len(tokens)):
                vec[i] = word2idx.get(tokens[i], 1)
            data.append(vec)
    return np.array(data)

def load_data(mode):
    try:
        source_word2idx, source_idx2word = load_vocabs(params.source_vocab_file, params.min_count)
        target_word2idx, target_idx2word = load_vocabs(params.target_vocab_file, params.min_count)
    except:
        create_vocabs([params.train_source_file, params.test_source_file], params.source_vocab_file)
        create_vocabs([params.train_target_file, params.test_target_file], params.target_vocab_file)
        source_word2idx, source_idx2word = load_vocabs(params.source_vocab_file, params.min_count)
        target_word2idx, target_idx2word = load_vocabs(params.target_vocab_file, params.min_count)
    if mode == "train":
        source_inputs = create_inputs(params.train_source_file, source_word2idx, params.max_len)
        target_inputs = create_inputs(params.train_target_file, target_word2idx, params.max_len, add_start=True)
    else:
        source_inputs = create_inputs(params.test_source_file, source_word2idx, params.max_len)
        target_inputs = create_inputs(params.test_target_file, target_word2idx, params.max_len, add_start=True)
    source_vocab_size = len(source_word2idx.keys())
    target_vocab_size = len(target_word2idx.keys())
    return source_inputs, target_inputs, source_vocab_size, target_vocab_size, target_word2idx, target_idx2word

def get_batches(source_inputs, target_inputs, shuffle=True):
    idx = np.arange(source_inputs.shape[0])
    if shuffle:
        np.random.shuffle(idx)
    batches = [
        idx[range(i * params.batch_size, (i + 1) * params.batch_size)] for i in range(int(len(idx) / params.batch_size))
    ]
    for i in batches:
        yield (source_inputs[i], target_inputs[i])


