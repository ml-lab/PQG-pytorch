import torch
import torch.nn as nn
import copy
import numpy as np
from misc.net_utils import one_hot
import misc.net_utils as net_utils

class Rollout(object):

    def __init__(self, e_model, g_model, vocab_size, update_rate):

        self.e_ori_model = e_model
        self.g_ori_model = g_model
        self.e_own_model = copy.deepcopy(e_model)
        self.g_own_model = copy.deepcopy(g_model)
        self.update_rate = update_rate
        self.vocab_size = vocab_size

    def get_reward(self, x, num, discriminator, z):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        # rewards = torch.zeros()
        with torch.no_grad():
            batch_size = x.size(0)
            seq_len = x.size(1)
            data = torch.zeros(batch_size * seq_len, seq_len, device=x.device, dtype=torch.long)
            rewards = torch.zeros(batch_size * seq_len, device=x.device)
            y = torch.cat([z for i in range(seq_len)], dim=0)
            for l in range(seq_len):
                data[l*batch_size:(l+1)*batch_size, :l+1] = x[:,:l+1]

            for i in range(num):
                # for l in range(1, seq_len):
                #     data = x[:, 0:l]
                #     samples = net_utils.prob2pred(self.g_own_model(self.e_own_model(one_hot(data, self.vocab_size)), teacher_forcing=False))
                #     pred = discriminator(samples)
                #     pred = pred.cpu().data.numpy()
                #     if i == 0:
                #         rewards.append(pred)
                #     else:
                #         rewards[l-1] += pred    
                # for the last token
                # pred = discriminator(x)
                # pred = pred.cpu().data.numpy()
                # if i == 0:
                #     rewards.append(pred)
                # else:
                #     rewards[seq_len-1] += pred
                '''forcing changed ----------------'''
                rewards += discriminator(net_utils.prob2pred(self.g_own_model(self.e_own_model(one_hot(data, self.vocab_size)), true_out=y)), y).view(-1)
                
            rewards = rewards / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.e_ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.e_own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
        
        for name, param in self.g_ori_model.module.named_parameters():
            dic[name] = param.data
        for name, param in self.g_own_model.module.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]