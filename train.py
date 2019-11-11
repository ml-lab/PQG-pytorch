import torch
import torch.nn as nn
import torch.optim as optim
from misc.LanguageModel import layer as LanguageModel
import misc.utils as utils
import misc.net_utils as net_utils
from misc.FixedGRU import FixedGRU
from misc.HybridCNNLong import HybridCNNLong as DocumentCNN
from model import Model
from pycocoevalcap.eval import COCOEvalCap
from tensorboardX import SummaryWriter
import subprocess
import torch.utils.data as Data
from misc.dataloader import Dataloader
import time
import os
import math
from rollout import Rollout
import gc
from discriminator import Discriminator

parser = utils.make_parser()
args = parser.parse_args()

torch.manual_seed(args.seed)

log_folder = 'logs'
save_folder = 'save'
sample_folder = 'samples'


if args.start_from != 'None':
    folder = args.start_from.split('/')[-2]
else :
    folder = time.strftime("%d-%m-%Y_%H:%M:%S")
    folder = args.name + folder
    subprocess.run(['mkdir', os.path.join(save_folder, folder)])
    subprocess.run(['mkdir', os.path.join(sample_folder, folder)])

file_sample = os.path.join('samples', folder, 'samples')

import itertools

writer_train = SummaryWriter(os.path.join(log_folder, folder + 'train'))
writer_val = SummaryWriter(os.path.join(log_folder, folder + 'val'))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get dataloader
data = Dataloader(args.input_json, args.input_ques_h5)

train_loader = Data.DataLoader(Data.Subset(data, range(args.train_dataset_len)), batch_size = args.batch_size, shuffle=True)
test_loader = Data.DataLoader(Data.Subset(data, range(args.train_dataset_len, args.train_dataset_len + args.val_dataset_len)), batch_size = args.batch_size, shuffle=True)
test_loader_iter = itertools.cycle(test_loader)

iter_per_epoch = (args.train_dataset_len + args.batch_size - 1)/ args.batch_size


def getObjsForScores(real_sents, pred_sents):
    class coco:

        def __init__(self, sents):
            self.sents = sents
            self.imgToAnns = [[{'caption' : sents[i]}] for i in range(len(sents))]

        def getImgIds(self):
            return [i for i in range(len(self.sents))]


    return coco(real_sents), coco(pred_sents)

def eval_batch(encoder, generator, discriminator, test_loader_iter, writer_val, log_idx=0, sample_flag=False):
    
    encoder.eval()
    generator.eval()
    discriminator.eval()
    # with torch.no_grad():
    device = generator.module.device
    vocab_size = data.getVocabSize()
    seq, seq_len, sim_seq, sim_seq_len, _ = next(test_loader_iter)
    seq, seq_len, sim_seq, sim_seq_len = seq.to(device), seq_len.to(device), sim_seq.to(device), sim_seq_len.to(device)

    prob_seq = generator(encoder(net_utils.one_hot(seq, vocab_size)), teacher_forcing=False)
    pred_seq = net_utils.prob2pred(prob_seq)
    # local loss criterion
    loss_f = nn.CrossEntropyLoss()

    # compute local loss
    local_loss = loss_f(prob_seq.permute(0, 2, 1), sim_seq) # sim_seq or seq
    
    binary_loss = nn.BCELoss()

    pred_real = discriminator(pred_seq) 

    pred_real_real = discriminator(sim_seq) # sim_seq or seq
    
    g_loss = binary_loss(pred_real, discriminator.module.out_tensor(pred_real, 'real'))
    d_loss = binary_loss(pred_real_real, discriminator.module.out_tensor(pred_real_real, 'real')) + binary_loss(pred_real, discriminator.module.out_tensor(pred_real, 'fake'))

    writer_val.add_scalar('local_loss', local_loss.item(), log_idx)
    writer_val.add_scalar('g_loss_cross_entropy', g_loss.item() , log_idx)
    writer_val.add_scalar('d_loss', d_loss.item(), log_idx)
    
    if sample_flag:
        pred_seq = pred_seq.long()
        sents = net_utils.decode_sequence(data.ix_to_word, pred_seq)
        inp_sents = net_utils.decode_sequence(data.ix_to_word, seq)
        sim_out_sents = net_utils.decode_sequence(data.ix_to_word, sim_seq)
        
        coco, cocoRes = getObjsForScores(sim_out_sents, sents)

        evalObj = COCOEvalCap(coco, cocoRes)

        evalObj.evaluate()

        for key in evalObj.eval:
            writer_val.add_scalar(key, evalObj.eval[key], log_idx)

        f_sample = open(file_sample + str(log_idx) + '.txt', 'w')
        
        idx = 1
        for r, s, t in zip(inp_sents, sim_out_sents, sents):

            f_sample.write(str(idx) + '\ninp : ' + r + '\nout : ' + s + '\npred : ' + t + '\n\n')
            idx += 1

        f_sample.close()
    # if it == iter_per_epoch - 1:
    #     save_model(model, epoch, it, local_loss, g_loss, d_loss)
    torch.cuda.empty_cache()

def save_model(model, epoch, it, local_loss, g_loss, d_loss):

    PATH = os.path.join(save_folder, folder, str(epoch) + '_' + str(it) + '.tar')
    
    checkpoint = {
        'epoch' : epoch,
        'iter' : it,
        'encoder_state_dict' : model.encoder.state_dict(),
        'generator_state_dict' : model.generator.state_dict(),
        'discriminator_state_dict' : model.discriminator.state_dict(), 
        'local_loss' : local_loss, 
        'g_loss' : g_loss,
        'd_loss' : d_loss
    }

    torch.save(checkpoint, PATH)

def pre_epoch_training(encoder, generator, discriminator, e_optim, g_optim, d_optim, dataloader, writer_train, g_epoch=1, d_epoch=1):

    encoder.train()
    generator.train()
    discriminator.train()
    loss_f = nn.NLLLoss(reduction='mean')
    loss_f = loss_f.to(generator.device)
    device = generator.device
    vocab_size = data.getVocabSize()
    generator = nn.DataParallel(generator)
    print('Pre training Generator ...')

    for epoch in range(g_epoch):
        idx = 0
        for seq, seq_len, sim_seq, sim_seq_len, _ in dataloader:

            seq, seq_len, sim_seq, sim_seq_len = seq.to(device), seq_len.to(device), sim_seq.to(device), sim_seq_len.to(device)
            seq_one_hot = net_utils.one_hot(seq, vocab_size)
            encoded_seq = encoder(seq_one_hot)
            print(gc.collect(), end='-')
            prob_sim_seq = generator(encoded_seq, true_out=sim_seq)
            g_loss = loss_f(prob_sim_seq.permute(0, 2, 1), sim_seq)
            e_optim.zero_grad()
            g_optim.zero_grad()
            g_loss.backward()
            e_optim.step()
            g_optim.step()
            idx += 1
            # writer_train.add_scalar('pre_train_generator_NLL_loss', g_loss, idx)
    
            torch.cuda.empty_cache()
            print(gc.collect(), end=' ')

    print('Pre training Discriminator ...')
    d_loss_f = nn.BCELoss()
    d_loss_f = d_loss_f.to(device)
    for epoch in range(d_epoch):
        idx = 0
        for seq, seq_len, sim_seq, sim_seq_len, _ in dataloader:
            
            seq, seq_len, sim_seq, sim_seq_len = seq.to(device), seq_len.to(device), sim_seq.to(device), sim_seq_len.to(device)
            seq_one_hot = net_utils.one_hot(seq, vocab_size)
            encoded_seq = encoder(seq_one_hot)
            print(gc.collect(), end='-')
            torch.cuda.empty_cache()
            prob_sim_seq = generator(encoded_seq, true_out=sim_seq)
            pred_sim_seq = net_utils.prob2pred(prob_sim_seq)
            d_fake = discriminator(pred_sim_seq)
            d_real = discriminator(sim_seq)
            d_loss = d_loss_f(d_fake, discriminator.out_tensor(d_fake, 'fake')) + d_loss_f(d_real, discriminator.out_tensor(d_real, 'real'))
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            idx += 1
            # writer_train.add_scalar('pre_train_discriminator_BCE_loss', d_loss, idx)
            torch.cuda.empty_cache()
            print(gc.collect(), end=' ')

def train_epoch(encoder, generator, discriminator, e_optim, g_optim, d_optim, rollout, train_loader, test_loader, writer_train, writer_val, log_idx=0, log_per_iter=args.log_every):

    encoder.train()
    generator.train()
    discriminator.train()
    d_loss_f = nn.BCELoss()
    d_loss_f = d_loss_f.to(generator.device)
    device = generator.device
    vocab_size = data.getVocabSize()
    batch_size = args.batch_size
    idx = 0
    loss_f = nn.CrossEntropyLoss()
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    g_loss = 0
    d_loss = 0
    local_loss = 0

    for batch in train_loader:
        
        if idx == 0:
            print('Generator training ...')
        # get new batch
        seq, seq_len, sim_seq, sim_seq_len, _ = batch
        seq, seq_len, sim_seq, sim_seq_len = seq.to(device), seq_len.to(device), sim_seq.to(device), sim_seq_len.to(device)
        
        seq_one_hot = net_utils.one_hot(seq, vocab_size)

        encoded = encoder(seq_one_hot)
        prob_sim_seq_l = generator(encoded, teacher_forcing=False)
        local_loss = loss_f(prob_sim_seq_l.permute(0, 2, 1), sim_seq)
        pred_sim_seq = net_utils.prob2pred(prob_sim_seq_l)
        rewards = torch.Tensor(rollout.get_reward(pred_sim_seq, args.roll, discriminator))
        print(rewards.size(), end='||', flush=True)
        rewards = torch.exp(rewards).contiguous().view((-1, ))
        rewards = rewards.to(device)
        print(rewards.size(), end='|', flush=True)

        prob_sim_seq = generator(encoder(net_utils.one_hot(pred_sim_seq, vocab_size)), true_out=pred_sim_seq)
        g_loss = gan_loss_f(prob_sim_seq, pred_sim_seq.contiguous().view((-1)), rewards) / batch_size
        g_optim.zero_grad()
        e_optim.zero_grad()
        (g_loss + local_loss).backward()
        e_optim.step()
        g_optim.step()

        rollout.update_params()
        
        if idx == 0:
            print('Discriminator training ...')
            
        seq_one_hot = net_utils.one_hot(seq, vocab_size)
        encoded_seq = encoder(seq_one_hot)
        prob_sim_seq = generator(encoded_seq, teacher_forcing=False)
        pred_sim_seq = net_utils.prob2pred(prob_sim_seq)
        d_fake = discriminator(pred_sim_seq)
        d_real = discriminator(sim_seq) # sim_seq OR seq ???
        d_loss = d_loss_f(d_fake, discriminator.module.out_tensor(d_fake, 'fake')) + d_loss_f(d_real, discriminator.module.out_tensor(d_real, 'real'))
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()    
        torch.cuda.empty_cache()
        print(idx, end='-', flush=True)
        idx += 1

    writer_train.add_scalar('g_loss', g_loss.item(), log_idx)
    writer_train.add_scalar('local_loss', local_loss.item(), log_idx)
    writer_train.add_scalar('d_loss', d_loss.item(), log_idx)
    eval_batch(encoder, generator, discriminator, test_loader_iter, writer_val, log_idx = log_idx,sample_flag=True)
    log_idx+=1

    return log_idx

def gan_loss_f(prob, target, reward):
    """
    Args:
        prob: (N, C), torch Variable
        target : (N, ), torch Variable
        reward : (N, ), torch Variable
    """
    N = target.size(0)
    C = prob.size(-1)
    prob = prob.view(-1, prob.size(-1))
    one_hot_x = net_utils.one_hot(target, C)
    one_hot_x = one_hot_x.type(torch.ByteTensor)
    one_hot_x = one_hot_x.to(prob.device)
    loss = torch.masked_select(prob, one_hot_x)
    loss = loss * reward
    loss =  -torch.sum(loss)
    return loss

if __name__ == '__main__' :

    # make model
    encoder = DocumentCNN(data.getVocabSize(), args.txtSize, dropout=args.drop_prob_lm, avg=1, cnn_dim=args.cnn_dim)
    generator = LanguageModel(args.input_encoding_size, args.rnn_size, data.getSeqLength(), data.getVocabSize(), num_layers=args.rnn_layers, dropout=args.drop_prob_lm)
    discriminator = Discriminator(encoder, args.cnn_dim, data.getVocabSize(), dropout=args.drop_prob_lm)

    # decay_factor = math.exp(math.log(0.1) / (1500 * 1250))
    lr = 0.0008
    e_optim = optim.RMSprop(encoder.parameters(), lr=lr)
    g_optim = optim.RMSprop(generator.parameters(), lr=lr)
    d_optim = optim.RMSprop(discriminator.parameters(), lr=lr)

    if args.start_from != 'None':

        print('loading model from ' + args.start_from)
        checkpoint = torch.load(args.start_from, map_location=torch.device('cpu'))
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # lr = 0.0008 * (decay_factor ** (start_epoch))
        # for opt in model.get_opt():
        #     for g in opt.param_groups:
        #         g['lr'] = lr
        #     print("learning rate = ", lr)
    else:
        start_epoch = 0


    # sheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=1, gamma=decay_factor)

    encoder = encoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    n_epoch = args.n_epoch

    encoder.train()
    generator.train()
    discriminator.train()

    print('Pre epoch training of generator and discriminator...')
    pre_epoch_training(encoder, generator, discriminator, e_optim, g_optim, d_optim, train_loader, writer_train)

    print('Roll model ...')

    rollout = Rollout(encoder, nn.DataParallel(generator), data.getVocabSize(), 0.8)

    print('Adversarial training starts ...')
    log_idx = 0

    for epoch in range(start_epoch, start_epoch + n_epoch):

        log_idx = train_epoch(encoder, generator, discriminator, e_optim, g_optim, d_optim, rollout, train_loader, test_loader_iter, writer_train, writer_val, log_idx=log_idx)
        
        print('epoch = ', epoch, flush=True)

    print('Done !!!')