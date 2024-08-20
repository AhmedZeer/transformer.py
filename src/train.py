import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import causal_mask_creator, BiDataset
from model import transformer_builder
from config import generate_config, get_weight_file_path

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
    data = load_dataset("Helsinki-NLP/opus-100", f"{config['lang_src']}-{config['lang_trg']}", split = 'train')
    src = get_from_ds(data, config['lang_src'])
    tgt = get_from_ds(data, config['lang_trg'])

    train_size = int(len(data) * 0.9)
    eval_size  = len(data) - train_size

    src_tokenizer = get_or_build_tokenizer(config, data, config['lang_src'])
    trg_tokenizer = get_or_build_tokenizer(config, data, config['lang_trg'])

    train_data_raw, eval_data_raw = random_split(data, [train_size, eval_size])

    train_data = BiDataset(src_tokenizer, trg_tokenizer,train_data_raw,config['seq_len'],config['lang_trg'],config['lang_src'])
    valid_data = BiDataset(src_tokenizer, trg_tokenizer,eval_data_raw,config['seq_len'],config['lang_trg'],config['lang_src'])

    max_src_seq = 0
    max_trg_seq = 0
    for item in data:
        src_text_seq = src_tokenizer.encode(item[config['lang_src']]).ids
        trg_text_seq = src_tokenizer.encode(item[config['lang_trg']]).ids
        max_src_seq = max(max_src_seq, len(src_text_seq))
        max_trg_seq = max(max_trg_seq, len(trg_text_seq))

    print(f"Max Source Seq Len: {max_src_seq}")
    print(f"Max Target Seq Len: {max_trg_seq}")

    train_dl = DataLoader(train_data, config['batch_size'], shuffle = True)
    valid_dl = DataLoader(valid_data, config['batch_size'], shuffle = True)

    return train_dl, valid_dl, src_tokenizer, trg_tokenizer

def get_model(config, src_vocab_size, tgt_vocab_size):
    return transformer_builder(config['seq_len'], config['seq_len'],
                               src_vocab_size, tgt_vocab_size)



def train(config):

    device = torch.device('cuda' if torch.coda.is_available() else 'cpu')

    train_dl, valid_dl, src_tokenizer, trg_tokenizer = get_ds(config)
    model = get_model(config, src_tokenizer.vocab_size, trg_tokenizer.vocab_size)

    optimizer = torch.optim.Adam(model.parametrs(), config['lr'])
    if config['preload']:
        model_path = get_weight_file_path(config, config['num_epochs'])
        print("Preloading Model")
        state = torch.load(model_path)












