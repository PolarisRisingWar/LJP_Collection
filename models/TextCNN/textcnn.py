import json,wandb,thulac
import pickle as pk
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


device="cuda:1"
Cutter=thulac.thulac(seg_only=True)

ds=256
max_length=512
dc=256
learning_rate=1e-3
window_sizes=[2,3,4,5]
filter_num=64
dropout_rate=0.5
batch_size=128
epoch_num=16

#dataset-specific超参：
law_num=
charge_num=
penalty_num=
embedding_dim=200

wandb.init(
    project="",
    
    config={
    "model":"TextCNN",
    "learning_rate":learning_rate
    }
)

####数据预处理
def load_w2v_matrix(numpy_path:str,w2id_path:str):
    """
    输入预训练模型路径和预训练模型类型
    输出numpy.ndarray格式的矩阵[词数,词向量维度] 和 word2id词典
    """
    with open(w2id_path,'rb') as f:
        word2id_dict=pk.load(f)
        f.close()
    
    word_embedding=torch.from_numpy(np.cast[np.float32](np.load(numpy_path)))

    return (word_embedding,word2id_dict)

w,d=load_w2v_matrix("cail_thulac.npy",
                    "w2id_thulac.pkl")

def transform_word2id(word):
    if not (word in d.keys()):
        return d["BLANK"]
    else:
        return d[word]

def parse_one_case(case:str):
    result=[]
    sentence=case.strip().split()
    for word in sentence:
        if len(word) == 0:
            continue
        if word in [",", ".", "?", "\"", "”", "。", "？", "","，",",","、","”"]:
            continue
        result.append(transform_word2id(word))
    return result

def word_list2tensor(word_list:list[list]):
    """将一个batch转换为一个张量"""
    word_num=min(max(len(case) for case in word_list),max_length)
    # 初始化一个全0的张量
    tensor=torch.zeros(len(word_list),word_num,dtype=torch.long)

    # 填充张量
    for sent_idx, sent in enumerate(word_list):
        for word_idx, word in enumerate(sent):
            if word_idx < word_num:  # 检查词索引是否小于最大词数
                tensor[sent_idx, word_idx] = word

    return tensor




class DataAdapterDataset(Dataset):
    """案例事实文本和标签对"""
    def __init__(self,mode:str) -> None:
        #略
    
    def __getitem__(self, index):
        return [self.fact[index],self.article[index],self.charge[index],self.penalty_label[index]]
    
    def __len__(self):
        return len(self.fact)

def collate_fn(batch):
    facts=[parse_one_case(b[0]) for b in batch]
    
    #将单词转换为张量
    fact_tensor=word_list2tensor(facts)

    law_labels=torch.tensor([b[1] for b in batch])
    charge_labels=torch.tensor([b[2] for b in batch])
    penalty_labels=torch.tensor([b[3] for b in batch])

    return [fact_tensor,law_labels,charge_labels,penalty_labels]

dataloader = DataLoader(DataAdapterDataset('train'), batch_size=batch_size, shuffle=True, num_workers=0,
                        drop_last=False,collate_fn=collate_fn)

test_dataloader = DataLoader(DataAdapterDataset('test'), batch_size = batch_size, shuffle=False,num_workers=0,
                        drop_last=False,collate_fn=collate_fn)



####模型构建
class CNNEncoder(nn.Module):
    """https://github.com/PolarisRisingWar/pytorch_text_classification/blob/master/pycls/models/textcnn.py"""
    def __init__(self):
        super(CNNEncoder,self).__init__()

        self.convs=nn.ModuleList([nn.Conv2d(1,filter_num,(k,embedding_dim)) \
                                  for k in window_sizes])
    
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  #[batch_size,num_filters,H_{in}-1]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  #[batch_size, num_filters]
        return x

    def forward(self,x):
        x=x.unsqueeze(1)  #[batch_size, 1, max_sentence_length, embedding_dim]
        x=torch.cat([self.conv_and_pool(x,conv) for conv in self.convs], 1)  #[batch_size, len(filter_sizes)*num_filters]

        return x




        


class TextCNN(nn.Module):
    def __init__(self,embedding:np.array,dropout_rate:float=dropout_rate):
        super(TextCNN,self).__init__()

        self.embs = nn.Embedding(164673, 200)
        self.embs.weight.data.copy_(embedding)
        self.embs.weight.requires_grad=False

        self.encoder=CNNEncoder()
        self.dropout=nn.Dropout(dropout_rate)

        self.decoder1=nn.Linear(len(window_sizes)*filter_num,law_num)
        self.decoder2=nn.Linear(len(window_sizes)*filter_num,charge_num)
        self.decoder3=nn.Linear(len(window_sizes)*filter_num,penalty_num)

    def forward(self,x):
        x=self.embs(x)
        x=self.encoder(x)
        x=self.dropout(x)

        return (self.decoder1(x),self.decoder2(x),self.decoder3(x))


model=TextCNN(w)
model=model.to(device)



criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)



#训练
for epoch in range(epoch_num):
    #训练
    for step,batch in enumerate(tqdm(dataloader)):
        model.train()
        optimizer.zero_grad()

        facts_embedding=batch[0].to(device)
        law_labels=batch[1].to(device)
        charge_labels=batch[2].to(device)
        penalty_labels=batch[3].to(device)

        o=model(facts_embedding)

        loss1=criterion(o[0],law_labels)
        loss2=criterion(o[1],charge_labels)
        loss3=criterion(o[2],penalty_labels)
        loss=loss1+loss2+loss3
        
        wandb.log({"loss":loss,"law_loss":loss1,"charge_loss":loss2,"penalty_loss":loss3})
        loss.backward()
        optimizer.step()

#测试
law_label=[]
law_predict=[]
charge_label=[]
charge_predict=[]
penalty_label=[]
penalty_predict=[]
with torch.no_grad():
    for step,batch in enumerate(tqdm(test_dataloader)):
        model.eval()

        facts_embedding=batch[0].to(device)

        o=model(facts_embedding)

        law_label.extend([i.item() for i in batch[1]])
        law_predict.extend([i.item() for i in torch.argmax(o[0],1)])
        charge_label.extend([i.item() for i in batch[2]])
        charge_predict.extend([i.item() for i in torch.argmax(o[1],1)])
        penalty_label.extend([i.item() for i in batch[3]])
        penalty_predict.extend([i.item() for i in torch.argmax(o[2],1)])



wandb.log({"acc_law":accuracy_score(law_label,law_predict),
           "p_law":precision_score(law_label,law_predict,average='macro'),
           "r_law":recall_score(law_label,law_predict,average='macro'),
           "f1_law":f1_score(law_label,law_predict,average='macro'),
           "acc_charge":accuracy_score(charge_label,charge_predict),
           "p_charge":precision_score(charge_label,charge_predict,average='macro'),
           "r_charge":recall_score(charge_label,charge_predict,average='macro'),
           "f1_charge":f1_score(charge_label,charge_predict,average='macro'),
           "acc_penalty":accuracy_score(penalty_label,penalty_predict),
           "p_penalty":precision_score(penalty_label,penalty_predict,average='macro'),
           "r_penalty":recall_score(penalty_label,penalty_predict,average='macro'),
           "f1_penalty":f1_score(penalty_label,penalty_predict,average='macro')})

wandb.finish()