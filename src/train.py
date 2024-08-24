import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import causal_mask_creator, BiDataset
from model import transformer_builder
from config import generate_config, get_weight_file_path

from tqdm import tqdm

def get_from_ds(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[SOS]', '[EOS]', '[PAD]', '[UNK]'])
        tokenizer.train_from_iterator(get_from_ds(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    data = load_dataset("Helsinki-NLP/opus-100", f"{config['lang_src']}-{config['lang_trg']}", split = 'train')

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
        src_text_seq = src_tokenizer.encode(item['translation'][config['lang_src']]).ids
        trg_text_seq = src_tokenizer.encode(item['translation'][config['lang_trg']]).ids
        max_src_seq = max(max_src_seq, len(src_text_seq))
        max_trg_seq = max(max_trg_seq, len(trg_text_seq))

    print(f"Max Source Seq Len: {max_src_seq}")
    print(f"Max Target Seq Len: {max_trg_seq}")

    train_dl = DataLoader(train_data, config['batch_size'], shuffle = True)
    valid_dl = DataLoader(valid_data, config['batch_size'], shuffle = True)

    return train_dl, valid_dl, src_tokenizer, trg_tokenizer

def get_model(config, src_vocab_size, tgt_vocab_size):
    return transformer_builder(config['seq_len'], src_vocab_size, tgt_vocab_size)



def train(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device Being Used: {device}")

    train_dl, valid_dl, src_tokenizer, trg_tokenizer = get_ds(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), trg_tokenizer.get_vocab_size()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])

    writer = SummaryWriter(config['experiment_name'])
    # Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    initial_epoch = 0
    global_step   = 0
    if config['preload']:
        model_path = get_weight_file_path(config, config['preload'])
        print("Preloading Model")
        state = torch.load(model_path)
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        optimizer.load_state_dict(state['optimizer_state_dict'])
    else:
        print("No Model To Reload.")

    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for e in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dl, desc=f"Epoch: {e:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # B x seq_len
            decoder_input = batch['decoder_input'].to(device) # B x seq_len
            encoder_mask  = batch['encoder_mask'].to(device)  # B x 1 x seq_len
            decoder_mask  = batch['decoder_mask'].to(device)  # B x seq_len x seq_len
            label = batch['label'].to(device)                 # B x seq_len

            encoder_output = model.encode(encoder_input, encoder_mask)

            assert encoder_output != None, "Train: Encoder Output Is None."

            decoder_output = model.decode(decoder_input, encoder_output,
                                          encoder_mask, decoder_mask)
            pred = model.project(decoder_output) # B x seq_len x vocab_size

            # (B, Seq, vocab_size) -> (B * Seq, vocab_size)
            pred  = pred.view(-1, src_tokenizer.get_vocab_size() )
            # (B, Seq) -> (B * Seq)
            label = label.view(-1)

            loss = loss_fn(pred, label)
            batch_iterator.set_postfix({"Loss: ":f"{loss.item()}"})

            writer.add_scalar("Loss",loss.item(),global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        model_filename = get_weight_file_path(config, f"{e:02d}")
        torch.save(
            {
                "global_step":global_step,
                "optimizer_state_dict":optimizer.state_dict(),
                "model_state_dict":model.state_dict(),
                "epoch":e
            }, model_filename
        )

if __name__ == '__main__':
    config = generate_config()
    train(config)
