import numpy as np
import joblib

dictionary=joblib.load(r'data/directory.pkl')
fets=['gender','icu_type','ethnicity','apache_2_bodysystem','apache_3j_bodysystem']


def ordinal(input_val,feats):
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value


def __boBinary(x,y):
    addTo=y-len(bin(x)[2:])
    return '0'*addTo+bin(x)[2:]

def process(values):
    temp=[]
    for fet in fets:
        
        if fet=='icu_type':
            bnNo=__boBinary(dictionary[fet][values[0]],4)
            for i in bnNo:
                temp.append(np.int8(i))
        elif fet=='ethnicity':
            bnNo=__boBinary(dictionary[fet][values[1]],3)
            for i in bnNo:
                temp.append(np.int8(i))
        elif fet=='apache_2_bodysystem':
            bnNo=__boBinary(dictionary[fet][values[2]],4)
            for i in bnNo:
                temp.append(np.int8(i))
        elif fet=='apache_3j_bodysystem':
            bnNo=__boBinary(dictionary[fet][values[3]],4)
            for i in bnNo:
                temp.append(np.int8(i))
        
    return temp

def get_prediction(data,model):
    pred=model.predict(data)
    return pred

