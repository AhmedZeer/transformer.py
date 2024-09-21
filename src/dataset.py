import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tokenizer import *

class train_data(Dataset):
    def __init__(self, data, tgt_tokenizer, src_tokenizer, seq_len):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.seq_len = seq_len

    def __getitem__(self, idx):

        src = torch.tensor(self.src_tokenizer.encode(self.data[idx]['en']).ids)
        tgt = torch.tensor(self.tgt_tokenizer.encode(self.data[idx]['tr']).ids)

        sos_id = torch.tensor(self.src_tokenizer.token_to_id("[SOS]"), dtype = torch.int64).unsqueeze(0)
        eos_id = torch.tensor(self.src_tokenizer.token_to_id("[EOS]"), dtype = torch.int64).unsqueeze(0)
        pad_id = torch.tensor(self.src_tokenizer.token_to_id("[PAD]"), dtype = torch.int64)

        # [SOS] - src - [EOS] - [PAD]
        src_pads_num = self.seq_len - len(src) - 2

        # src - [EOS] - [PAD]
        tgt_pads_num = self.seq_len - len(tgt) - 1

        assert tgt_pads_num > 0, f"Short Seq Len. SeqLen: {self.seq_len} | {len(tgt) - 1} -> {tgt_pads_num}"
        assert src_pads_num > 0, f"Short Seq Len. SeqLen: {self.seq_len} | {len(src) - 2} -> {src_pads_num}"

        # [SOS] and [EOS]
        src = torch.cat(
            [
                sos_id,
                src,
                eos_id,
                torch.tensor([pad_id] * src_pads_num)
            ]
        )

        # Only [EOS]
        label = torch.cat(
            [
                tgt,
                eos_id,
                torch.tensor([pad_id] * tgt_pads_num)
            ]
        )

        # Only [SOS]
        tgt = torch.cat(
            [
                sos_id,
                tgt,
                torch.tensor([pad_id] * tgt_pads_num)
            ]
        )


        src_msk = src == pad_id
        tgt_msk = tgt == pad_id

        return dict(
            source_seq = src,
            target_seq = tgt,
            label = label,
            padding_source_mask = src_msk,
            padding_target_mask = tgt_msk,
        )

    def __len__(self):
        return len(self.data)


def create_train_dataloader(tgt_toknizer, src_toknizer, seq_len, batch_size):

    dataset = load_dataset("Helsinki-NLP/opus-100", "en-tr", split="train[:2000]")

    raw_data = []
    for item in dataset['translation']:
        tgt = item['tr']
        src = item['en']
        if len(tgt) < seq_len and len(src) < seq_len:
            raw_data.append(item)

    train_dataset = train_data(raw_data, tgt_toknizer, src_toknizer, seq_len)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    return train_dl
