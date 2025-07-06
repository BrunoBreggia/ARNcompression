from typing import Callable

import torch
from torch import nn

EOS = "eos"

class Codificador(nn.Module):

    def __init__(self, alphabet_size, max_length, hidden_units, hid_layers):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.max_length = max_length
        self.hidden_units = hidden_units
        self.hid_layers = hid_layers

        self.encoder_gru = nn.GRU(alphabet_size, hidden_units, hid_layers, batch_first=True)
        self.encoder_linear = nn.Linear(hidden_units, alphabet_size)
        # self.activation = nn.Sigmoid()

        self.decoder_gru = nn.GRU(alphabet_size, hidden_units, hid_layers, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_units, alphabet_size)
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        # initializations
        inner_buffer = torch.zeros((self.max_length, self.alphabet_size))
        hidd_state = torch.zeros((self.hidden_units, self.hid_layers))
        # cell_state = torch.zeros((self.hidden_units, self.hid_layers))

        output, hidd_state = self.encoder_lstm(x, hidd_state)
        y = self.encoder_linear(output[:,-1,:])
        y = nn.functional.sigmoid(y)

        for i in range(self.max_length):
            output = self.encoder(output)
            if output == EOS:
                break
            self.inner_buffer[]


