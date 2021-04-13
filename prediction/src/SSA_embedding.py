from __future__ import print_function,division

import sys
import os
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

import pandas as pd
import torch
import torch.nn as nn
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

from prediction.preprocessing.alphabets import Uniprot21
 
import warnings
 
warnings.filterwarnings('ignore')

def unstack_lstm(lstm):
    device = next(iter(lstm.parameters())).device

    in_size = lstm.input_size
    hidden_dim = lstm.hidden_size
    layers = []
    for i in range(lstm.num_layers):
        layer = nn.LSTM(in_size, hidden_dim, batch_first=True, bidirectional=True)
        layer.to(device)

        attributes = ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l']
        for attr in attributes:
            dest = attr + '0'
            src = attr + str(i)
            getattr(layer, dest).data[:] = getattr(lstm, src)
             
            dest = attr + '0_reverse'
            src = attr + str(i) + '_reverse'
            getattr(layer, dest).data[:] = getattr(lstm, src)
             
        layer.flatten_parameters()
        layers.append(layer)
        in_size = 2*hidden_dim
    return layers
def embed_stack(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=False):
    zs = []
    
    x_onehot = x.new(x.size(0),x.size(1), 21).float().zero_()
    x_onehot.scatter_(2,x.unsqueeze(2),1)
    zs.append(x_onehot)
    
    h = lm_embed(x)
    if include_lm and not final_only:
        zs.append(h)

    if lstm_stack is not None:
        for lstm in lstm_stack:
            h,_ = lstm(h)
            if not final_only:
                zs.append(h)
    if proj is not None:
        h = proj(h.squeeze(0)).unsqueeze(0)
        zs.append(h)

    z = torch.cat(zs, 2)
    return z
def embed_sequence(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=False
                  ,  pool='none', use_cuda=False):

    if len(x) == 0:
        return None

    alphabet = Uniprot21()
    x = x.upper()
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        x = x.to(DEVICE)
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = embed_stack(x, lm_embed, lstm_stack, proj
                       , include_lm=include_lm, final_only=final_only)
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        z = z.cpu().numpy()

    return z
def load_model(path, use_cuda=False):
     
    encoder = torch.load(path)
    encoder.eval()
    if use_cuda:
        encoder=encoder.to(DEVICE)
    encoder = encoder.embedding

    lm_embed = encoder.embed
    lstm_stack = unstack_lstm(encoder.rnn)
    proj = encoder.proj

    return lm_embed, lstm_stack, proj

def BiLSTM_Embed(input_seq):
    T0=time.time()
    BiLSTMEMB_=[]
    PID=[]
    print("\nBiLSTM Embedding...")
    print("Loading BiLSTM Model...", file=sys.stderr, end='\r')
    lm_embed, lstm_stack, proj = load_model("./prediction/embbed_models/SSA_embed.model", use_cuda=True)
    proj = None

    for key,value in input_seq.items():
        PID.append(key)
        sequence = value.encode("utf-8")
        z = embed_sequence(sequence, lm_embed, lstm_stack, proj
                                  ,  final_only=False,include_lm = True
                                  , pool='avg', use_cuda=True)
        BiLSTMEMB_.append(z)
    bilstm_feature=pd.DataFrame(BiLSTMEMB_)
    col=["BiLSTM_F"+str(i+1) for i in range(0,3605)]
    bilstm_feature.columns=col
    bilstm_feature=pd.concat([bilstm_feature],axis=1)
    bilstm_feature.index=PID
    bilstm_feature.to_csv("./dataset/bilstm_feature.csv")
    print("BiLSTM embedding finished@@￥￥￥￥￥")
    print("it took %0.3f mins.\n"%((time.time()-T0)/60))

    return bilstm_feature

    
    
     
    




