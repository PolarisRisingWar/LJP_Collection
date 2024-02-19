import wandb,thulac
import pickle as pk
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import xavier_normal_

torch.autograd.set_detect_anomaly(True)

device="cuda:1"
Cutter=thulac.thulac(seg_only=True)

#原论文中的超参：
ds=256
max_length=512
dc=256
learning_rate=1e-3
window_sizes=[2,3,4,5]
filter_num=64
dropout_rate=0.5
batch_size=128
epoch_num=16

#dataset-specific超参
law_num=
charge_num=
penalty_num=
embedding_dim=200

wandb.init(
    project="",
    
    config={
    "model":"MPBFN",
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



class MPBFNDecoder(nn.Module):
    def __init__(self):
        super(MPBFNDecoder,self).__init__()
        self.S1=nn.Parameter(xavier_normal_(torch.empty(law_num,ds)))
        self.S2=nn.Parameter(xavier_normal_(torch.empty(charge_num,ds)))
        self.S3=nn.Parameter(xavier_normal_(torch.empty(penalty_num,ds)))

        self.linear1=nn.Linear(len(window_sizes)*filter_num,law_num)

        self.Ws1=nn.Parameter(xavier_normal_(torch.empty(dc,ds)))
        self.Ws2=nn.Parameter(xavier_normal_(torch.empty(dc,ds)))
        self.Ws3=nn.Parameter(xavier_normal_(torch.empty(dc,ds)))

        self.Wf12=nn.Parameter(xavier_normal_(torch.empty(charge_num,ds)))
        self.Wf13=nn.Parameter(xavier_normal_(torch.empty(penalty_num,ds)))
        self.Wf23=nn.Parameter(xavier_normal_(torch.empty(penalty_num,ds)))

        self.b12=nn.Parameter(torch.zeros(charge_num,))
        self.b13=nn.Parameter(torch.zeros(penalty_num,))
        self.b23=nn.Parameter(torch.zeros(penalty_num,))

        self.Wg21=nn.Parameter(xavier_normal_(torch.empty(law_num,ds)))
        self.Wg31=nn.Parameter(xavier_normal_(torch.empty(law_num,ds)))
        self.Wg32=nn.Parameter(xavier_normal_(torch.empty(charge_num,ds)))

        self.b21=nn.Parameter(torch.zeros(law_num,))
        self.b31=nn.Parameter(torch.zeros(law_num,))
        self.b32=nn.Parameter(torch.zeros(charge_num,))

        self.elu=nn.ELU()
        self.softmax=nn.Softmax(1)
        self.sigmoid=nn.Sigmoid()
    
    def normalize(self,x,epsilon=1e-8):
        return x / (torch.sum(x, dim=-1, keepdim=True) + epsilon)

    def forward(self,factori):
        res1=self.linear1(factori)
        res1=self.softmax(res1)  #公式9

        lsv1=torch.matmul(res1,self.S1)  #公式4
        
        sem1=torch.matmul(self.Ws1,lsv1.unsqueeze(2))
        sem1=self.elu(sem1)  #公式5

        fact1=torch.mul(factori,sem1.squeeze(2))  #公式6

        pred12=torch.matmul(self.Wf12,fact1.unsqueeze(2)).squeeze(2)+self.b12
        pred12=self.softmax(pred12)  #公式7

        pred13=torch.matmul(self.Wf13,fact1.unsqueeze(2)).squeeze(2)+self.b13
        pred13=self.softmax(pred13)

        res2=self.normalize(pred12)  #公式9

        lsv2=torch.matmul(res2,self.S2).unsqueeze(2)  #公式4

        gate21=torch.matmul(self.Wg21,lsv2).squeeze(2)+self.b21
        gate21=self.sigmoid(gate21)  #公式8

        sem2=torch.matmul(self.Ws2,lsv2)
        sem2=self.elu(sem2)  #公式5

        fact2=torch.mul(factori,sem2.squeeze(2))  #公式6

        pred23=torch.matmul(self.Wf23,fact2.unsqueeze(2)).squeeze(2)+self.b23
        pred23=self.softmax(pred23)

        res3=torch.mul(pred13,pred23)
        res3=self.normalize(res3)

        lsv3=torch.matmul(res3,self.S3).unsqueeze(2)  #公式4

        gate31=torch.matmul(self.Wg31,lsv3).squeeze(2)+self.b31
        gate31=self.sigmoid(gate31)  #公式8

        gate32=torch.matmul(self.Wg32,lsv3).squeeze(2)+self.b32
        gate32=self.sigmoid(gate32)  #公式8

        ver1=self.normalize(torch.mul(gate21,gate31))  #公式10

        y1=torch.mul(res1,ver1)

        ver2=self.normalize(gate32)

        y2=torch.mul(res2,ver2)
        
        return (y1,y2,res3)

        


class MPBFN(nn.Module):
    def __init__(self,embedding:np.array,dropout_rate:float=dropout_rate):
        super(MPBFN,self).__init__()

        self.embs = nn.Embedding(164673, 200)
        self.embs.weight.data.copy_(embedding)
        self.embs.weight.requires_grad=False

        self.encoder=CNNEncoder()
        self.dropout=nn.Dropout(dropout_rate)

        self.decoder=MPBFNDecoder()

    def forward(self,x):
        x=self.embs(x)
        x=self.encoder(x)
        x=self.dropout(x)
        x=self.decoder(x)

        return x


model=MPBFN(w)
model=model.to(device)



criterion=nn.NLLLoss()
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

        loss1=criterion(torch.log(o[0] + 1e-8),law_labels)
        loss2=criterion(torch.log(o[1] + 1e-8),charge_labels)
        loss3=criterion(torch.log(o[2] + 1e-8),penalty_labels)
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