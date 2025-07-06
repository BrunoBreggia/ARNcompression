"""
Implementación del código del paper:
"Sequence to Sequence Learning with Neural Networks"
de Sutskever, Vinyals and Le

Guiado por el tutorial: https://www.youtube.com/watch?v=EoGUlvhRYpk
de Aladdin Persson
"""

import torch
from numba.roc.gcn_occupancy import max_group_size
from ray.experimental.state.common import SummaryApiResponse
from ray.train import load_checkpoint, save_checkpoint
from torch import nn
from torch import optim
from torchtext.datasewt import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

spacy_ger = spacy.load("de")  # german tokenizer object
spacy_eng = spacy.load("en")  # english tokenizer object

def tokenizer_german(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_english(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenizer_german, lower=True,
               init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_english, lower=True,
               init_token='<sos>', eos_token='<eos>')

train_dataset, val_dataset, test_dataset = Multi30k.splits(exts=('.de', '.en'),
                                                           fields=(german, english))
german.build_vocab(train_dataset, max_size=10_000, min_freq=2)
english.build_vocab(train_dataset, max_size=10_000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p_dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p_dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p_dropout)

    def forward(self, x):
        # x shape: (seq_length, batch_size N)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p_dropout):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p_dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p_dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # shape of x: (N) but we want (1,N), one word at a time
        x = x.unsqueeze(0)

        embedding =self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # output shape: (1,N,hidden_size)

        predictions = self.linear(outputs)
        # predictions shape: (1,N,lenght_of_vocab)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, taecher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output
            # (n, eng_vocab_size)

            best_guess = output.argmax(i)

            # teacher forcing
            x = target[t] if random.random() < taecher_force_ratio else best_guess

        return outputs

# Training hyper-parameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyper-parameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f"runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, val_dataset, test_dataset),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                      hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.ptar"), model, optimizer)

for epoch in range(num_epochs):
    print(f"Epoch [{epoch} / {num_epochs}]")

    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)
        # output shape: (trg_len, batch_size, output_dim)

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar("Training loss", loss.item(), global_step=step)
        step += 1
