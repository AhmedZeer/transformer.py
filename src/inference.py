from numpy._core.defchararray import decode
import tokenizers
from config import *
import torch
from dataset import causal_mask_creator
from model import transformer_builder, Transformer
from tokenizers import Tokenizer
from train import get_ds, get_from_ds, get_model

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

config = generate_config()

def build_tokenizer(config, ds, lang):

    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[SOS]', '[EOS]', '[PAD]', '[UNK]'])
        tokenizer.train_from_iterator(get_from_ds(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    return tokenizer

def train_tokenizer():
    data = load_dataset("Helsinki-NLP/opus-100", f"{config['lang_src']}-{config['lang_trg']}", split = 'train')
    src_tokenizer = build_tokenizer(config, data, config['lang_src'])
    trg_tokenizer = build_tokenizer(config, data, config['lang_trg'])
    return src_tokenizer, trg_tokenizer

def greedy_decoder(model : Transformer,
                   encoder_output,
                   src_mask,
                   tgt_tokenizer : Tokenizer,
                   device, config):

    sos = tgt_tokenizer.token_to_id('[SOS]')
    eos = tgt_tokenizer.token_to_id('[EOS]')
    decoder_input = torch.empty([1,1], dtype = torch.int64).fill_(sos).to(device)
    # decoder_input, tgt_mask = padding(decoder_input, tgt_tokenizer, config, 'decoder')

    #def forward(self, x, encoder_output, src_mask, tgt_mask):
    while True:

        if len(decoder_input) >= config['seq_len']:
            break

        tgt_mask = causal_mask_creator(decoder_input.size(1))
        decoder_output = model.decode(decoder_input, encoder_output, src_mask, tgt_mask)
        # print("Dynamic Seq Len  :", decoder_input.shape)
        # print("Dynamic Tgt Mask :", tgt_mask.shape)

        # (B x Seq x Embed) -> In each batch, only project the latest sequence.
        # -> (B x Vocab)
        probas = model.project(decoder_output[:, -1])

        # ( B x Vocab ) -> Get the largest proba in each batch.
        _,next_word = torch.max(probas, dim = 1)
        next_word.squeeze_(0)

        if next_word == eos:
            break

        # Concatenate Each Batch With Its Next Word.
        decoder_input = torch.cat(
            [decoder_input,
            torch.empty(1,1).fill_(next_word).type_as(decoder_input).to(device)],
            dim = 1
        )


def padding(tokenized_query, src_tokenizer, config, mode='encoder'):

    sos = torch.tensor([src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
    eos = torch.tensor([src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
    pad = torch.tensor([src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    if mode == 'decoder':
        paddings = config['seq_len'] - len(tokenized_query)
        padded_sequence = torch.cat(
            [
                torch.tensor(tokenized_query, dtype = torch.int64),
                torch.tensor([[pad]] * paddings, dtype = torch.int64)
            ]
        )
        mask = (padded_sequence != pad).int().unsqueeze(0)

        # print("Encoder Padded Len:", len(padded_sequence))
        assert len(padded_sequence) == config['seq_len'], "Not Padded Well."
        return padded_sequence.transpose(1,0), mask

    else:
        paddings = config['seq_len'] - len(tokenized_query) - 2
        padded_sequence = torch.cat(
            [
                sos,
                torch.tensor(tokenized_query, dtype = torch.int64),
                eos,
                torch.tensor([pad] * paddings, dtype = torch.int64)
            ]
        )

        mask = (padded_sequence != pad).int().unsqueeze(0).unsqueeze(0)
        # print("Encoder Padded Len:", len(padded_sequence))

        assert len(padded_sequence) == config['seq_len'], "Not Padded Well."
        return padded_sequence, mask


def inference(model : Transformer, query : str,
              src_tokenizer : Tokenizer,
              tgt_tokenizer : Tokenizer, config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenized_query = src_tokenizer.encode(query).ids
    tokenized_query, src_mask = padding(tokenized_query, src_tokenizer, config)
    encoder_output = model.encode(tokenized_query, src_mask)

    greedy_decoder(model, encoder_output,
                   src_mask,
                   tgt_tokenizer, device, config)

if __name__ == '__main__':
    src_tokenizer, tgt_tokenizer = train_tokenizer()
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size())
    # 7
    inference(model, "HELLO THIS IS REALLY LONG SENTENCE AND SHOULD BE AAAAAAA", src_tokenizer, tgt_tokenizer, config)
