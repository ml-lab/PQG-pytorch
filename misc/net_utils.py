import misc.utils as utils
import torch

def decode_sequence(ix_to_word, seq):
    N, D = seq.size()[0], seq.size()[1]
    out = []
    EOS_flag = False
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if int(ix.item()) not in ix_to_word:
                print("UNK token ", str(ix.item()))
                word = ix_to_word[len(ix_to_word) - 1]
            else:
                word = ix_to_word[int(ix.item())]
            
            if word == '<PAD>':
                word = ''
            
            if j > 0:
                txt = txt + ' '
            txt = txt + word
        out.append(txt)
    return out

def clone_list(lst):
    new = []
    for t in lst:
        new.append(t)
    return new

def language_eval(predictions, id):
    out_struct = {
        "val_predictions": predictions
    }
    utils.write_json('coco-caption/val'+id+'.json', out_struct)
    import subprocess
    subprocess.run(['./misc/call_python_caption_eval.sh', 'val' + id + '.json'])
    result_struct = utils.read_json('coco-caption/val'+id+'.json_out.json')
    return result_struct

def one_hot(x, c):
    return torch.zeros(*x.size(), c, device=x.device).scatter_(-1, x.unsqueeze(-1), 1)

def prob2pred(prob):

    return torch.multinomial(torch.exp(prob.view(-1, prob.size(-1))), 1).view(prob.size(0), prob.size(1))