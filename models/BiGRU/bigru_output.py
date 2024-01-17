#Bi-GRU文本分类（用输出作为最终表征）在LADAN预处理后CAIL2018数据集上的实现
#词向量来自NeurJudge作者

import os,argparse,json
import pickle as pk
from datetime import datetime

import numpy as np

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder')  #数据集文件夹
parser.add_argument('--data_type')  #small / big
args = parser.parse_args()

gpu_device='cuda:1'
epoch_num=8
train_batch_size=16
inference_batch_size=768
max_sentence_length=512
learning_rate=1e-3
dropout_rate=0
hidden_dim=150
layer_num=1
embedding_dim=200

if args.data_type=='small':
    charge_num=119
    law_num=103
else:
    charge_num=130
    law_num=118

print(datetime.now())



###导入word2vec矩阵
word2id=json.load(open('word2vec/word2id.json'))
word2vec=json.load(open('word2vec/word2vec.json'))
embedding=np.zeros((339503,embedding_dim))
for k in word2id:
    if k in word2vec:
        embedding[word2id[k],:]=[float(factor) for factor in word2vec[k]]
embedding[1,:]=np.mean(embedding[2:,:],axis=0).tolist()  #用平均值来做UNK
embedding=torch.from_numpy(embedding)



###搭建数据集
class DataAdapterDataset(Dataset):
    """案例事实文本和标签对"""
    def __init__(self,mode='train') -> None:
        data=pk.load(open(os.path.join(args.data_folder,mode+'_processed_thulac_Legal_basis.pkl'),'rb'))
        self.fact=data['fact']
        self.article=data['law_label_lists']
        self.charge=data['accu_label_lists']
        self.penalty_label=data['term_lists']
    
    def __getitem__(self, index):
        return [self.fact[index],self.article[index],self.charge[index],self.penalty_label[index]]
    
    def __len__(self):
        return len(self.fact)

train_dataloader=DataLoader(DataAdapterDataset('train'),batch_size=train_batch_size,shuffle=True)
test_dataloader=DataLoader(DataAdapterDataset('test'),batch_size=inference_batch_size,shuffle=False)

def process_data(batch):
    #对每个batch的数据进行处理，转换为pad后的Tensor、sent_len（每个样本的词数）和标签
    original_text=[sentence.split() for sentence in batch[0]]
    sent_len_max=max([len(s) for s in original_text])
    sent_len_max=min(sent_len_max,max_sentence_length)
    padded_tensor=torch.zeros((len(original_text),sent_len_max),dtype=torch.int)
    sent_len=[]
    for (i,sentence) in enumerate(original_text):
        sent_len.append(min(max_sentence_length,len(sentence)))
        for (j,word) in enumerate(sentence):
            if j>=max_sentence_length:
                break
            padded_tensor[i,j]=word2id[word] if word in word2id else word2id['UNK']

    return (padded_tensor,sent_len,batch[1],batch[2],batch[3])



#建模
class GRUEncoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,dropout_rate,bias=True,bidirectional=True):
        super(GRUEncoder,self).__init__()

        self.embs=nn.Embedding(339503,input_dim)
        self.embs.weight.data.copy_(embedding)
        self.embs.weight.requires_grad=False

        self.rnns=nn.GRU(input_size=input_dim,hidden_size=hidden_dim,num_layers=num_layers,bias=bias,dropout=dropout_rate,bidirectional=bidirectional,
                        batch_first=True)
        self.lin1=nn.Linear(in_features=hidden_dim*2 if bidirectional else hidden_dim,out_features=law_num)
        self.lin2=nn.Linear(in_features=hidden_dim*2 if bidirectional else hidden_dim,out_features=charge_num)
        self.lin3=nn.Linear(in_features=hidden_dim*2 if bidirectional else hidden_dim,out_features=11)

    def forward(self,x,sent_len):
        x=self.embs(x)
        packed_input=nn.utils.rnn.pack_padded_sequence(x,lengths=sent_len,batch_first=True,enforce_sorted=False)
        op,hn=self.rnns(packed_input)
        op,lens_unpacked=nn.utils.rnn.pad_packed_sequence(op,batch_first=True)

        #[batch_size,max_sequence_length,hidden_dim*num_directions]
        #取最后一个有效时间步上的表征，作为最终表征
        outputs=op[torch.arange(0,op.size()[0]).to(gpu_device),lens_unpacked-1]
        return (self.lin1(outputs),self.lin2(outputs),self.lin3(outputs))


model=GRUEncoder(input_dim=embedding_dim,num_layers=layer_num,dropout_rate=dropout_rate,
                hidden_dim=hidden_dim)
model.to(gpu_device)

optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate)
loss_func=torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(epoch_num):
    #训练
    for batch in train_dataloader:
        optimizer.zero_grad()
        (x,sent_len,law_label,charge_label,penalty_label)=process_data(batch)
        outputs=model(x.to(gpu_device),sent_len)
        train_loss=loss_func(outputs[0],law_label.to(gpu_device))+loss_func(outputs[1],charge_label.to(gpu_device))+\
                    loss_func(outputs[2],penalty_label.to(gpu_device))
        train_loss.backward()
        optimizer.step()
    
    print('epoch'+str(epoch)+'训练完毕！当前时间是：'+str(datetime.now()))


with torch.no_grad():
    #测试
    model.eval()
    law_label=[]
    law_predict=[]
    charge_label=[]
    charge_predict=[]
    penalty_label=[]
    penalty_predict=[]
    for batch in test_dataloader:
        (x,sent_len,law_label_batch,charge_label_batch,penalty_label_batch)=process_data(batch)
        outputs=model(x.to(gpu_device),sent_len)
        law_label.extend([i.item() for i in law_label_batch])
        law_predict.extend([i.item() for i in torch.argmax(outputs[0],1)])
        charge_label.extend([i.item() for i in charge_label_batch])
        charge_predict.extend([i.item() for i in torch.argmax(outputs[1],1)])
        penalty_label.extend([i.item() for i in penalty_label_batch])
        penalty_predict.extend([i.item() for i in torch.argmax(outputs[2],1)])
    print(accuracy_score(law_label,law_predict))
    print(precision_score(law_label,law_predict,average='macro'))
    print(recall_score(law_label,law_predict,average='macro'))
    print(f1_score(law_label,law_predict,average='macro'))
    print(accuracy_score(charge_label,charge_predict))
    print(precision_score(charge_label,charge_predict,average='macro'))
    print(recall_score(charge_label,charge_predict,average='macro'))
    print(f1_score(charge_label,charge_predict,average='macro'))
    print(accuracy_score(penalty_label,penalty_predict))
    print(precision_score(penalty_label,penalty_predict,average='macro'))
    print(recall_score(penalty_label,penalty_predict,average='macro'))
    print(f1_score(penalty_label,penalty_predict,average='macro'))

print('代码运行结束！当前时间是：'+str(datetime.now()))