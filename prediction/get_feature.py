from __future__ import print_function,division
import sys
sys.path.append('./prediction/')
from src.SSA_embedding import BiLSTM_Embed
from src.UniRep_emb import UniRep_Embed
import pandas as pd
from Bio import SeqIO

def fasta(data):
    seq_dict={}
    id_list=[]
    seq_list=[]
    for seq_record in SeqIO.parse(data, "fasta"):
        id= seq_record.id
        seq=seq_record.seq
        seq_dict[id]=seq
        id_list.append(id)
        seq_list.append(seq)
    return seq_dict,id_list,seq_list

def GnerateFeatures(seq_dict):
    feature_BiLSTM = BiLSTM_Embed(seq_dict)
    feature_Unirep = UniRep_Embed(seq_dict)
    fusedFeature = pd.concat([feature_BiLSTM,feature_Unirep], axis=1)
    fusedFeature.to_csv("./dataset/fusedFeature.csv")
    feature_id= pd.read_csv("./prediction/feature_id.csv")
    LGB_ALL_K = fusedFeature[feature_id.columns]
    LGB_ALL_K.index = fusedFeature.index
    LGB_ALL_K.to_csv("./dataset/selection_feature.csv")
    print("Selected Top %d features" % (LGB_ALL_K.shape[1] - 1))
    print("LGBoosting features selections completed!!!!\n\n")
    return LGB_ALL_K
