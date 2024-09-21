from tokenizers import Tokenizer
from pathlib import Path
import torch

def get_tokenizer(lang):

    path = f"./tokenizer_{lang}.json"
    assert Path.exists(Path(path)), f"Tokenizer path not found, {path}"

    return Tokenizer.from_file(path)

def tokenize_input(sentence:str, tokenizer:Tokenizer):
    return torch.tensor(tokenizer.encode(sentence).ids, dtype=torch.int64)
