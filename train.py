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

def eval_batch(model, test_loader_iter, epoch, it, writer_val, sample_flag=False):
    
    with torch.no_grad():
        if it == iter_per_epoch - 1:
            sample_flag == True
        seq, seq_len, sim_seq, sim_seq_len, _ = next(test_loader_iter)
        seq, seq_len, sim_seq, sim_seq_len = seq.to(model.device), seq_len.to(model.device), sim_seq.to(model.device), sim_seq_len.to(model.device)

        prob_seq = model.generator.sample(model.encoder(net_utils.one_hot(seq, model.vocab_size)))
        pred_seq = model.prob2pred(prob_seq)
        # local loss criterion
        loss_f = nn.CrossEntropyLoss()

        # compute local loss
        local_loss = loss_f(prob_seq.permute(0, 2, 1), seq) # sim_seq or seq
        
        binary_loss = nn.BCELoss()

        pred_real = model.discriminator(pred_seq) 

        pred_real_real = model.discriminator(sim_seq) # sim_seq or seq
        
        g_loss = binary_loss(pred_real, model.discriminator.out_tensor(pred_real, 'real'))
        d_loss = binary_loss(pred_real_real, model.discriminator.out_tensor(pred_real_real, 'real')) + binary_loss(pred_real, model.discriminator.out_tensor(pred_real, 'fake'))

        writer_val.add_scalar('local_loss', local_loss.item(), epoch * iter_per_epoch + it)
        writer_val.add_scalar('g_loss_cross_entropy', g_loss.item() , epoch * iter_per_epoch + it)
        writer_val.add_scalar('d_loss', d_loss.item(), epoch * iter_per_epoch + it)
        
        if sample_flag:
            pred_seq = pred_seq.long()
            sents = net_utils.decode_sequence(data.ix_to_word, pred_seq)
            inp_sents = net_utils.decode_sequence(data.ix_to_word, seq)
            sim_out_sents = net_utils.decode_sequence(data.ix_to_word, sim_seq)
            
            coco, cocoRes = getObjsForScores(sim_out_sents, sents)

            evalObj = COCOEvalCap(coco, cocoRes)

            evalObj.evaluate()

            for key in evalObj.eval:
                writer_val.add_scalar(key, evalObj.eval[key], epoch * iter_per_epoch + it)

            f_sample = open(file_sample + str(epoch) + '_' + str(it) + '.txt', 'w')
            
            idx = 1
            for r, s, t in zip(inp_sents, sim_out_sents, sents):

                f_sample.write(str(idx) + '\ninp : ' + r + '\nout : ' + s + '\npred : ' + t + '\n\n')
                idx += 1

            f_sample.close()
        if it == iter_per_epoch - 1:
            save_model(model, epoch, it, local_loss, g_loss, d_loss)
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

def pre_epoch_training(model, dataloader, writer_train, g_epoch=1, d_epoch=1):

    model.train()
    loss_f = nn.NLLLoss(reduction='mean')
    loss_f = loss_f.to(model.device)

    print('Pre training Generator ...')

    for epoch in range(g_epoch):
        idx = 0
        for seq, seq_len, sim_seq, sim_seq_len, _ in dataloader:

            seq, seq_len, sim_seq, sim_seq_len = seq.to(model.device), seq_len.to(model.device), sim_seq.to(model.device), sim_seq_len.to(model.device)
            seq_one_hot = net_utils.one_hot(seq, model.vocab_size)
            encoded_seq = model.encoder(seq_one_hot)
            prob_sim_seq = model.generator(encoded_seq, seq, seq_len)
            g_loss = loss_f(prob_sim_seq.permute(0, 2, 1), seq)
            model.e_opt.zero_grad()
            model.g_opt.zero_grad()
            g_loss.backward()
            model.e_opt.step()
            model.g_opt.step()
            idx += 1
            # writer_train.add_scalar('pre_train_generator_NLL_loss', g_loss, idx)
    
            torch.cuda.empty_cache()
            gc.collect()
    print('Pre training Discriminator ...')
    d_loss_f = nn.BCELoss()
    d_loss_f = d_loss_f.to(model.device)
    for epoch in range(d_epoch):
        idx = 0
        for seq, seq_len, sim_seq, sim_seq_len, _ in dataloader:
            
            seq, seq_len, sim_seq, sim_seq_len = seq.to(model.device), seq_len.to(model.device), sim_seq.to(model.device), sim_seq_len.to(model.device)
            seq_one_hot = net_utils.one_hot(seq, model.vocab_size)
            encoded_seq = model.encoder(seq_one_hot)
            prob_sim_seq = model.generator.sample(encoded_seq)
            pred_sim_seq = model.prob2pred(prob_sim_seq)
            d_fake = model.discriminator(pred_sim_seq)
            d_real = model.discriminator(sim_seq) # sim_seq OR seq ???
            d_loss = d_loss_f(d_fake, model.discriminator.out_tensor(d_fake, 'fake')) + d_loss_f(d_real, model.discriminator.out_tensor(d_real, 'real'))
            model.d_opt.zero_grad()
            d_loss.backward()
            model.d_opt.step()
            idx += 1
            # writer_train.add_scalar('pre_train_discriminator_BCE_loss', d_loss, idx)
            torch.cuda.empty_cache()
            gc.collect()
def train_epoch(model, rollout, train_loader, test_loader, writer_train, writer_val, epoch=0, log_per_iter=args.log_every):

    model.train()
    d_loss_f = nn.BCELoss()
    d_loss_f = d_loss_f.to(model.device)

    idx = 0
    loss_f = nn.CrossEntropyLoss()
    for batch in train_loader:
        
        if idx == 0:
            print('Generator training ...')
        # get new batch
        seq, seq_len, sim_seq, sim_seq_len, _ = batch
        seq, seq_len, sim_seq, sim_seq_len = seq.to(model.device), seq_len.to(model.device), sim_seq.to(model.device), sim_seq_len.to(model.device)
        
        seq_one_hot = net_utils.one_hot(seq, model.vocab_size)

        encoded = model.encoder(seq_one_hot)
        prob_sim_seq_l = model.generator.sample(encoded)
        local_loss = loss_f(prob_sim_seq_l.permute(0, 2, 1), seq)
        pred_sim_seq = model.prob2pred(prob_sim_seq_l)
        rewards = torch.Tensor(rollout.get_reward(pred_sim_seq, args.roll, model.discriminator))
        rewards = torch.exp(rewards).contiguous().view((-1, ))
        rewards = rewards.to(model.device)

        prob_sim_seq = model.generator(model.encoder(net_utils.one_hot(pred_sim_seq, model.vocab_size)), pred_sim_seq)
        g_loss = gan_loss_f(prob_sim_seq, pred_sim_seq.contiguous().view((-1)), rewards) / model.batch_size
        model.g_opt.zero_grad()
        model.e_opt.zero_grad()
        (g_loss + local_loss).backward()
        model.e_opt.step()
        model.g_opt.step()
        writer_train.add_scalar('g_loss', g_loss.item(), epoch * iter_per_epoch + idx)
        writer_train.add_scalar('local_loss', local_loss.item(), epoch * iter_per_epoch + idx)

        rollout.update_params()
        
        if idx == 0:
            print('Discriminator training ...')
            
        seq_one_hot = net_utils.one_hot(seq, model.vocab_size)
        encoded_seq = model.encoder(seq_one_hot)
        prob_sim_seq = model.generator.sample(encoded_seq)
        pred_sim_seq = model.prob2pred(prob_sim_seq)
        d_fake = model.discriminator(pred_sim_seq)
        d_real = model.discriminator(sim_seq) # sim_seq OR seq ???
        d_loss = d_loss_f(d_fake, model.discriminator.out_tensor(d_fake, 'fake')) + d_loss_f(d_real, model.discriminator.out_tensor(d_real, 'real'))
        model.d_opt.zero_grad()
        d_loss.backward()
        model.d_opt.step()    
        writer_train.add_scalar('d_loss', d_loss.item(), idx)
        eval_batch(model, test_loader, epoch, idx, writer_val, sample_flag=((idx+1)%log_per_iter == 0))
        torch.cuda.empty_cache()
        idx += 1

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
    model = Model(args, data)

    # decay_factor = math.exp(math.log(0.1) / (1500 * 1250))

    model.make_opt(0.0008)

    if args.start_from != 'None':

        print('loading model from ' + args.start_from)
        checkpoint = torch.load(args.start_from, map_location=torch.device('cpu'))
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # lr = 0.0008 * (decay_factor ** (start_epoch))
        # for opt in model.get_opt():
        #     for g in opt.param_groups:
        #         g['lr'] = lr
        #     print("learning rate = ", lr)
    else:
        start_epoch = 0


    # sheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=1, gamma=decay_factor)

    model = model.to(device)

    n_epoch = args.n_epoch

    model.train()
    print('Pre epoch training of generator and discriminator...')
    # pre_epoch_training(model, train_loader, writer_train)

    print('Adversarial training starts ...')

    rollout = Rollout(model.encoder, model.generator, 0.8)

    for epoch in range(start_epoch, start_epoch + n_epoch):

        train_epoch(model, rollout, train_loader, test_loader_iter, writer_train, writer_val, epoch=epoch)
        # sheduler.step()

        # n_batch = data.getDataNum(1) // args.batch_size
        
        # local_loss = local_loss / n_batch
        # global_loss = global_loss / n_batch
        
        print('epoch = ', epoch)
        # eval_batch(model, device, epoch + 1, -1)

    print('Done !!!')
    