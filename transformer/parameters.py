# encoding: utf-8
import os
train_source_file = "../data/train/news-commentary-v12.de-en.de"
train_target_file = "../data/train/news-commentary-v12.de-en.en"
test_source_file = "../data/dev/newstest2013.de"
test_target_file = "../data/dev/newstest2013.en"
source_vocab_file = "../data/temp/vocab.de"
target_vocab_file = "../data/temp/vocab.en"

checkpoint_dir = os.path.join(os.getcwd(), "saved_model", "transformer")
model_name = "transformer_v1"

min_count = 20
max_len = 10
batch_size = 32
num_units = 128
num_heads = 8
num_blocks = 6
lr = 0.0001
num_epochs = 10000
dropout_rate = 0.1
is_training = True
sinusoid = False
