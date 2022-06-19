from deepsith import DeepSITH
import torch
from torch import nn as nn
import numpy as np

# Tensor Type. Use torch.cuda.FloatTensor to put all SITH math 
# on the GPU.
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
ttype = FloatTensor

sith_params1 = {"in_features":1, 
                "tau_min":1, "tau_max":25.0, 'buff_max':40,
                "k":84, 'dt':1, "ntau":15, 'g':.0,  
                "ttype":ttype, 'batch_norm':True,
                "hidden_size":35, "act_func":nn.ReLU()}
sith_params2 = {"in_features":sith_params1['hidden_size'], 
                "tau_min":1, "tau_max":100.0, 'buff_max':175,
                "k":40, 'dt':1, "ntau":15, 'g':.0, 
                "ttype":ttype, 'batch_norm':True,
                "hidden_size":35, "act_func":nn.ReLU()}
lp = [sith_params1, sith_params2]
deepsith_layers = DeepSITH(layer_params=lp, dropout=0.2)

class DeepSITH_Classifier(nn.Module):
    def __init__(self, out_features, layer_params, dropout=.5):
        super(DeepSITH_Classifier, self).__init__()
        last_hidden = layer_params[-1]['hidden_size']
        self.hs = DeepSITH(layer_params=layer_params, dropout=dropout)
        self.to_out = nn.Linear(last_hidden, out_features)
    def forward(self, inp):
        x = self.hs(inp)
        x = self.to_out(x)
        return x

def get_batch(batch_size, T, ttype):
    values = torch.rand(T, batch_size, requires_grad=False)
    indices = torch.zeros_like(values)
    half = int(T / 2)
    for i in range(batch_size):
        half_1 = np.random.randint(half)
        hals_2 = np.random.randint(half, T)
        indices[half_1, i] = 1
        indices[hals_2, i] = 1

    data = torch.stack((values, indices), dim=-1).type(ttype)
    targets = torch.mul(values, indices).sum(dim=0).type(ttype)
    return data, targets

def train(model, ttype, seq_length, optimizer, loss_func, 
          epoch, loss_buffer_size=20, batch_size=1, test_size=10,
          device='cuda', prog_bar=None):
    assert(loss_buffer_size%batch_size==0)

    losses = []
    perfs = []
    last_test_perf = 0
    for batch_idx in range(20000):
        model.train()
        sig, target = get_batch(batch_size, seq_length, ttype=ttype)
        sig = sig.transpose(0,1).transpose(1,2).unsqueeze(1)
        target = target.unsqueeze(1)
        optimizer.zero_grad()
        out = model(sig)
        loss = loss_func(out[:, -1, :],
                         target)
         
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        losses = losses[-loss_buffer_size:]
        if not (prog_bar is None):
            # Update progress_bar
            s = "{}:{} Loss: {:.8f}"
            format_list = [epoch, int(batch_idx/(50/batch_size)), np.mean(losses)]         
            s = s.format(*format_list)
            prog_bar.set_description(s)
        if ((batch_idx*batch_size)%loss_buffer_size == 0) & (batch_idx != 0):
            loss_track = {}
            #last_test_perf = test_norm(model, 'cuda', test_sig, test_class,
            #                                    batch_size=test_size, 
            #                                    )
            loss_track['avg_loss'] = np.mean(losses)
            #loss_track['last_test'] = last_test_perf
            loss_track['epoch'] = epoch
            loss_track['batch_idx'] = batch_idx

def test_norm(model, device, seq_length, loss_func, batch_size=100):
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        sig, target = get_batch(batch_size, seq_length, ttype=ttype)
        sig = sig.transpose(0,1).transpose(1,2).unsqueeze(1)
        target = target.unsqueeze(1)
        out = model(sig)
        loss = loss_func(out[:, -1, :],target)
    return loss

model = DeepSITH_Classifier(out_features=1,layer_params=lp, dropout=.0).cuda()