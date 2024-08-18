import torch
from torch.utils.data import Dataset

class BiLangDataset(Dataset):
    def __init__(self, ds, seq_len, src_lang, target_lang, src_tokenizer, tgt_tokenizer, ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.sos = torch.tensor([src_tokenizer.token_to_id(['[SOS]'])], dtype = torch.int64)
        self.eos = torch.tensor([src_tokenizer.token_to_id(['[EOS]'])], dtype = torch.int64)
        self.pad = torch.tensor([src_tokenizer.token_to_id(['[PAD]'])], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pair_data = self.ds[idx]

        src_data = pair_data['translation'][self.src_lang]
        target_data = pair_data['translation'][self.target_lang]
