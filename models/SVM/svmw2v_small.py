#用词向量的平均值池化作为样本表征，词向量来自NeurJudge项目

import argparse,json,os
import pickle as pk

from datetime import datetime

import numpy as np

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

print(datetime.now())

#命令行入参
parser=argparse.ArgumentParser()
parser.add_argument('--data_folder')
args=parser.parse_args()

word2vec=json.load(open('word2vec/word2vec.json'))
for k in word2vec:
    word2vec[k]=[float(factor) for factor in word2vec[k]]

f_train = pk.load(open(os.path.join(args.data_folder,'train_processed_thulac_Legal_basis.pkl'),'rb'))
f_valid = pk.load(open(os.path.join(args.data_folder,'valid_processed_thulac_Legal_basis.pkl'), 'rb'))
f_test = pk.load(open(os.path.join(args.data_folder,'test_processed_thulac_Legal_basis.pkl'), 'rb'))

#原word2vec中没有'UNK'，用所有表征的平均值来代替
word2vec['UNK']=np.mean(list(word2vec.values()),axis=0).tolist()

train_vectors=[[word2vec[word] if word in word2vec else word2vec['UNK'] for word in sentence.split(' ')] for sentence in f_train['fact']]
train_vector=np.array([np.mean(line,axis=0) for line in train_vectors])

test_vectors=[[word2vec[word] if word in word2vec else word2vec['UNK'] for word in sentence.split(' ')] for sentence in f_test['fact']]
test_vector=np.array([np.mean(line,axis=0) for line in test_vectors])

for label in ['law_label_lists', 'accu_label_lists', 'term_lists']:
    print(label)
    
    clf=make_pipeline(StandardScaler(),SVC())
    clf.fit(train_vector,f_train[label])

    y=f_test[label]
    predict_result=clf.predict(test_vector)
    print(metrics.accuracy_score(y,predict_result))
    print(metrics.precision_score(y,predict_result,average='macro'))
    print(metrics.recall_score(y,predict_result,average='macro'))
    print(metrics.f1_score(y,predict_result,average='macro'))


print(datetime.now())