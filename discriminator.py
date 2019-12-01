import torch
import torch.nn as nn
import misc.net_utils as net_utils

class Discriminator(nn.Module):

    def __init__(self, encoder, hidden_dim, vocab_size, dropout=0.55):
        super(Discriminator, self).__init__()
        '''self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = embedding
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        '''
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def forward(self, input, hidden=None):
        '''# input dim                                                # batch_size x seq_len
        emb = self.embeddings(input.long())                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        self.gru.flatten_parameters()
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out
        '''
        return torch.sigmoid(self.hidden2out(self.dropout_linear(self.encoder(net_utils.one_hot(input, self.vocab_size)))))
        # return net_utils.JointEmbeddingLoss(self.encoder(net_utils.one_hot(input, self.vocab_size)), self.encoder(net_utils.one_hot(out, self.vocab_size)))

        
    def out_tensor(self, inp, val):
        
        if val == 'real':
            return torch.ones(*inp.size(), device=inp.device)
        elif val == 'fake':
            return torch.zeros(*inp.size(), device=inp.device)
        else:
            raise ValueError