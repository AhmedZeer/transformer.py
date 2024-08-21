from numpy._core.defchararray import decode
import torch
from torch.utils.data import Dataset

class BiDataset( Dataset ):
    def __init__(self, src_tokenizer, trg_tokenizer,
                 ds, seq_len, trg_lang, src_lang):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.trg_lang = trg_lang 
        self.src_lang = src_lang
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        self.sos = torch.tensor([src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos = torch.tensor([src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad = torch.tensor([src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        pair_row = self.ds[index]
        trg_lang_pair = pair_row['translation'][self.trg_lang]
        src_lang_pair = pair_row['translation'][self.src_lang]

        encoded_src = self.src_tokenizer.encode(src_lang_pair).ids
        encoded_trg = self.trg_tokenizer.encode(trg_lang_pair).ids

        src_padding_num = self.seq_len - ( len(encoded_src) + 2 )
        trg_padding_num = self.seq_len - ( len(encoded_trg) + 1 )

        assert src_padding_num > 0, f"Too Short SeqLen, seq_len: {self.seq_len}, src_len:{len(encoded_src)}."
        assert trg_padding_num > 0, f"Too Short SeqLen, seq_len: {self.seq_len}, src_len:{len(encoded_src)}."

        encoder_input = torch.cat(
            [
                self.sos,
                torch.tensor(encoded_src,dtype=torch.int64),
                self.eos,
                torch.tensor([self.pad] * src_padding_num, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos,
                torch.tensor(encoded_trg, dtype = torch.int64),
                torch.tensor([self.pad] * trg_padding_num, dtype = torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(encoded_trg, dtype = torch.int64),
                self.eos,
                torch.tensor([self.pad] * trg_padding_num, dtype = torch.int64)
            ]
        )

        causal_mask = causal_mask_creator(decoder_input.shape[0])

        assert label.shape[0] == self.seq_len, f"Label Not Padded. seq_len:{self.seq_len}. Label_len:{label.shape[0]}."
        assert encoder_input.shape[0] == self.seq_len, f"Encoder_input Not Padded. Label_len:{encoder_input.shape[0]}."
        assert decoder_input.shape[0] == self.seq_len, f"Decoder_input Not Padded. Label_len:{encoder_input.shape[0]}."

        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            # (1, 1, seq_len)
            "encoder_mask" : (encoder_input != self.pad).int().unsqueeze(0).unsqueeze(0),
            # (1, seq_len) X (1, seq_len, seq_len)
            "decoder_mask" : (decoder_input != self.pad).int().unsqueeze(0) & causal_mask,
            "label":label, # (seq_len)
            "target_txt":trg_lang_pair,
            "src_txt:":src_lang_pair
        }

def causal_mask_creator(seq_len):
    return (torch.triu(torch.ones([seq_len, seq_len]), diagonal=1) == 0).unsqueeze(0).int()

