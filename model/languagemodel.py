import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LanguageModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, nlayers, dropout=0.5):
        super(LanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.hidden = None        
        self.drop = nn.Dropout(dropout) 
        self.encoder = nn.Embedding(vocab_size, embedding_dim)

        # The RNN takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, nlayers, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size, device):
        self.hidden = (torch.zeros(self.nlayers, batch_size, self.hidden_dim).to(device), 
                        torch.zeros(self.nlayers, batch_size, self.hidden_dim).to(device))

    def forward(self, data):
        emb = self.drop(self.encoder(data))
        output, hidden = self.rnn(emb, self.hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden
