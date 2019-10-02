import torch
import torch.nn as nn
from misc.LanguageModel import layer as LanguageModel
import misc.utils as utils
import misc.net_utils as net_utils
from misc.FixedGRU import FixedGRU
from misc.HybridCNNLong import HybridCNNLong as DocumentCNN
from discriminator import Discriminator
import torch.optim as optim
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Model:

    def __init__(self, args, dataloader):
        
        self.vocab_size = dataloader.getVocabSize()
        self.input_encoding_size = args.input_encoding_size
        self.rnn_size = args.rnn_size
        self.num_layers = args.rnn_layers
        self.drop_prob_lm = args.drop_prob_lm
        self.seq_length = dataloader.getSeqLength()
        self.batch_size = args.batch_size
        self.emb_size = args.input_encoding_size
        self.hidden_size = args.input_encoding_size
        self.att_size = args.att_size
        self.device = device
        
        self.encoder = DocumentCNN(self.vocab_size, args.txtSize, dropout=args.drop_prob_lm, avg=1, cnn_dim=args.cnn_dim)
        self.generator = LanguageModel(self.input_encoding_size, self.rnn_size, self.seq_length, self.vocab_size, num_layers=self.num_layers, dropout=self.drop_prob_lm)
        self.discriminator = Discriminator(self.emb_size, self.hidden_size, self.vocab_size, self.seq_length, self.generator.embedding, gpu=True, dropout=self.drop_prob_lm)

    def JointEmbeddingLoss(self, feature_emb1, feature_emb2):
        
        batch_size = feature_emb1.size()[0]
        loss = 0
        for i in range(batch_size):
            label_score = torch.dot(feature_emb1[i], feature_emb2[i])
            for j in range(batch_size):
                cur_score = torch.dot(feature_emb2[i], feature_emb1[j])
                score = cur_score - label_score + 1
                if 0 < score.item():
                    loss += max(0, cur_score - label_score + 1)

        denom = batch_size * batch_size
        
        return loss / denom

    def to(self, device):

        self.generator = self.generator.to(device)
        self.encoder = self.encoder.to(device)
        self.discriminator = self.discriminator.to(device)

        return self

    def train(self):
        
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()

        return self
    
    def eval(self):
        
        self.generator.eval()
        self.encoder.eval()
        self.discriminator.eval()

        return self
    
    def make_opt(self, lr, decay_factor=None, every_iter=None):

        self.e_opt = optim.RMSprop(self.encoder.parameters(), lr=lr)
        self.g_opt = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.d_opt = optim.RMSprop(self.discriminator.parameters(), lr=lr)        

    def prob2pred(self, prob):

        return self.generator.prob2pred(prob)