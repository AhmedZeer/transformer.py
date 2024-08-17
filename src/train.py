import torch
import torch.nn as nn
from torch.utils.data import random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

config = {}

def get_from_ds(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = config['tokenizer_path'].format(lang)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token=['[UNK]']))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[SOS]', '[EOS]', '[PAD]', '[UNK]'], min_freq = 3)
        tokenizer.train_from_iterator(get_from_ds(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    data = load_dataset("Helsinki-NLP/opus-100", f"{config['src_lang']}-{config['tgt_lang']}", split = 'train')
    src = get_from_ds(data, config['src_lang'])
    tgt = get_from_ds(data, config['tgt_lang'])

    train_size = int(len(data) * 0.9)
    eval_size = len(data) - train_size

    train_data_raw, eval_data = random_split(data, [train_size, eval_size])




