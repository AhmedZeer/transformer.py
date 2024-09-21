from tqdm import tqdm
import torch
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="transformer",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.01,
    "architecture": "transformer",
    "dataset": "en-tr",
    "epochs": 3,
    }
)

class Trainer():
    def __init__(self, transformer, train_dl, n_epochs, tokenizer, model_config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = transformer.to(self.device)
        self.train_dl = train_dl
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]')).to(self.device)
        self.loss = 0
        self.loss_per_epoch = []
        self.loss_per_batch = []
        self.model_config = model_config

    def _per_batch(self, data):
        self.optimizer.zero_grad()
        encoder_input, decoder_input, label = data['source_seq'], data['target_seq'], data['label']
        padding_source_mask, padding_target_mask = data['padding_source_mask'], data['padding_target_mask']
        encoder_input, decoder_input, label = encoder_input.to(self.device), decoder_input.to(self.device), label.to(self.device)
        padding_source_mask, padding_target_mask = padding_source_mask.to(self.device), padding_target_mask.to(self.device)
        y_hat = self.model(encoder_input, decoder_input, padding_source_mask, padding_target_mask)
        loss  = self.loss_fn(y_hat.view(-1, self.model_config['vocab_size']), label.view(-1))
        loss.backward()
        self.optimizer.step()
        self.loss += loss.item()
        self.loss_per_batch.append(loss.item())
        wandb.log({"loss_step":loss.item()})

    def _per_epoch(self):
        for data in tqdm(self.train_dl):
            # print("SourceShape:", data['source_seq'].shape)
            # print("TargetShape:", data['target_seq'].shape)
            # print("LabelShape:",  data['label'].shape)
            self._per_batch(data)
        self.loss /= len(self.train_dl)
        wandb.log({"loss_epoch":self.loss})
        self.loss_per_epoch.append(self.loss)

    def train(self, epochs):
        for epoch in tqdm(range(epochs)):
            self._per_epoch()
            print(f"EPOCH: {epoch} ---- LOSS: {self.loss}")

# trainer = Trainer(transformer, train_dl, encoder_config['n_epochs'], tr_tokenizer, encoder_config)

# trainer.train(10)
