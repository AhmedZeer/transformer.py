import torch
from torch.utils.data import Dataset

class BiDataset( Dataset ):
    def __init__(self, src_tokenizer, trg_tokenizer, d_model,
                 ds, seq_len, trg_lang, src_lang):
        super().__init__()
        self.ds = ds
        self.d_model = d_model
        self.seq_len = seq_len
        self.trg_lang = trg_lang 
        self.src_lang = src_lang
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        self.sos = torch.Tensor(src_tokenizer.token_to_id('[SOS]'))
        self.eos = torch.Tensor(src_tokenizer.token_to_id('[EOS]'))
        self.pad = torch.Tensor(src_tokenizer.token_to_id('[PAD]'))

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
        
        assert src_padding_num < 0, "Too Short SeqLen"
        assert trg_padding_num < 0, "Too Short SeqLen"

        encoder_input = torch.cat(
            [
                self.sos,
                torch.Tensor(encoded_src),
                self.eos, 
                torch.Tensor([self.pad for _ in range(src_padding_num)])
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos,
                torch.Tensor(encoded_trg),
                torch.Tensor([self.pad for _ in range(trg_padding_num)])
            ]
        )

        label = torch.cat(
            [
                self.sos,
                torch.Tensor(encoded_trg),
                self.eos,
                torch.Tensor([self.pad for _ in range(trg_padding_num)])
            ]
        )

        causal_mask = causal_mask_creator(label.shape[0])

        return {
            "encoder_input" : encoded_src,
            "decoder_input" : encoded_trg,
            "encoder_padding_mask" : (encoded_src != self.pad).unsqueeze(0).unsqueeze(0).int(),
            "decoder_padding_mask" : (encoded_trg != self.pad).unsqueeze(0).unsqueeze(0).int() & causal_mask,
            "label":label,
            "target_txt":trg_lang_pair,
            "src_txt:":src_lang_pair
        }

def causal_mask_creator(seq_len):
    return (torch.triu(torch.ones([seq_len, seq_len]), diagonal=1) == 0).unsqueeze(0).unsqueeze(0).int()

