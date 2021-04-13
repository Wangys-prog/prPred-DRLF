import joblib
import pandas as pd
from prediction.get_feature import fasta,GnerateFeatures
import argparse

def predict(inputfasta,outfile):
    seq_dict,id_list,seq_list = fasta(inputfasta)
    feature_sel= GnerateFeatures(seq_dict)
    model = joblib.load(open('./prediction/BiLSTM_unirep_model_lgb.pkl', 'rb'))
    x=pd.read_csv("./prediction/BiLSTM_unirep_lgbm_feature.csv")
    x2=x.iloc[:,1:]
    y=pd.read_csv("./prediction/train_label.csv")
    y2=y.iloc[:,1:]
    model.fit(x2,y2)
    pred_proba = model.predict_proba(feature_sel)
    id3=[]
    for i in range(len(id_list)):
        id2 = []
        id2.append(str(id_list[i]))
        id2.append(str(round(float(pred_proba[:,1][i]),3)))
        id3.append(id2)
    col = ["Sequence_ID", "R protein possibility"]
    result2=pd.DataFrame(data=id3,columns=col)
    result2.to_csv(outfile)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        'Script for predicting plant R protein using deep representation learning features')

    parser.add_argument('-i', type=str, help='input sequences in Fasta format')
    parser.add_argument('-o', type=str, help='path to saved CSV file')

    args = parser.parse_args()
    inputfasta= args.i
    outfile = args.o
    predict(inputfasta,outfile)

