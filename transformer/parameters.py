# encoding: utf-8
import os

# raw_source_file = "../data/train/train.zh"
# raw_target_file = "../data/train/train.en"
train_source_file = "../data/train/train1.zh"
train_target_file = "../data/train/train1.en"
test_source_file = "../data/train/test1.zh"
test_target_file = "../data/train/test1.en"
source_vocab_file = "../data/temp/vocab.zh"
target_vocab_file = "../data/temp/vocab.en"

checkpoint_dir = os.path.join(os.getcwd(), "saved_model", "transformer")
model_name = "transformer_v2"

min_count = 20
max_len = 20
batch_size = 32
num_units = 128
num_heads = 8
num_blocks = 6
num_epochs = 10000
dropout_rate = 0.1
is_training = True
sinusoid = False
