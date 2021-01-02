import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class SimpleRNN(BaseModel):
    def __init__(self, embedding_dim, hidden_dim, seq_len):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

        self.linear = nn.ModuleDict({
            'hidden': nn.Linear(self.hidden_dim, 1),
            'seq': nn.Linear(self.seq_len, 1)
        })

    def forward(self, x):
        batch_size = x.size(0)

        output, hidden = self.rnn(x)

        output = output.permute(0, 2, 1)
        output = self.linear['seq'](output)
        output = output.permute(0, 2, 1)

        hidden = hidden.permute(1, 0, 2)
        hidden += output
        hidden = self.linear['hidden'](hidden)

        return hidden.view(batch_size)


class GRU(BaseModel):
    def __init__(self, embedding_dim, hidden_dim, seq_len):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

        self.linear = nn.ModuleDict({
            'hidden': nn.Linear(self.hidden_dim, 1),
            'seq': nn.Linear(self.seq_len, 1)
        })

    def forward(self, x):
        batch_size = x.size(0)

        output, hidden = self.gru(x)

        output = output.permute(0, 2, 1)
        output = self.linear['seq'](output)
        output = output.permute(0, 2, 1)

        hidden = hidden.permute(1, 0, 2)
        hidden += output
        hidden = self.linear['hidden'](hidden)

        return hidden.view(batch_size)


class LSTM(BaseModel):
    def __init__(self, embedding_dim, hidden_dim, seq_len):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

        self.linear = nn.ModuleDict({
            'hidden': nn.Linear(self.hidden_dim, 1),
            'seq': nn.Linear(self.seq_len, 1)
        })

    def forward(self, x):
        batch_size = x.size(0)

        output, (hidden, cell) = self.lstm(x)

        output = output.permute(0, 2, 1)
        output = self.linear['seq'](output)
        output = output.permute(0, 2, 1)

        hidden = hidden.permute(1, 0, 2)
        hidden += output
        hidden = self.linear['hidden'](hidden)

        return hidden.view(batch_size)
