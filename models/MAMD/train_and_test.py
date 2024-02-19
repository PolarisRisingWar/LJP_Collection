import wandb
import pickle as pk
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset


device="cuda:2"
train_batch_size=128
test_batch_size=256
patience=10  #早停标准是accuracy
max_epoch=16
sigma=10  #sigma^2
epsilon=1e-6
learning_rate=1e-3

charge_num=

wandb.init(
    project="",
    
    config={
    "model":"MAMD",
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

def parse_one_case(sent):
    result = []
    sent = sent.strip().split()
    for word in sent:
        if len(word) == 0:
            continue
        if word in [",", ".", "?", "\"", "”", "。", "？", "","，",",","、","”"]:
            continue
        result.append(word)
    return result

def seq2tensor(fact:list,max_len=500):
    """
    输入列表格式的案件事实描述，每一个元素是一个案例的文本（已经经过分词，是一个用空格将词隔开的字符串）
    输出torch.Tensor和对应的长度（mask）
    """
    sents=[parse_one_case(x) for x in fact]
    sent_len_max = max([len(s) for s in sents])
    sent_len_max = min(sent_len_max, max_len)

    sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

    sent_len = torch.LongTensor(len(sents)).zero_()
    for s_id, sent in enumerate(sents):
        sent_len[s_id] = len(sent)
        for w_id, word in enumerate(sent):
            if w_id >= sent_len_max: break
            sent_tensor[s_id][w_id] = transform_word2id(word) 
    return sent_tensor,sent_len


class DataAdapterDataset(Dataset):
    """案例事实文本和标签对"""
    def __init__(self,mode:str) -> None:
        #略
    
    def __getitem__(self, index):
        return [self.fact[index],self.name[index],self.charge[index]]
    
    def __len__(self):
        return len(self.fact)

def find_all_occurrences(name:str,fact:str):
    start=0
    positions=[]
    facts=fact.split()
    while start<len(facts):
        if name.startswith(facts[start]):
            end=start
            while end<len(facts) and "".join(facts[start:end]) in name:
                if "".join(facts[start:end])==name:
                    positions.append(start)
                    break
                end+=1
        start+=1
    if len(positions)==0:
        positions=[0]
    return positions

def collate_fn(batch):
    facts=[b[0] for b in batch]
    fact_tensor=seq2tensor(facts)

    names=[b[1] for b in batch]
    name_index=[find_all_occurrences(names[i],facts[i]) for i in range(len(facts))]

    charge_labels=torch.tensor([b[2] for b in batch])
    
    sequence_length=fact_tensor[0].size()[1]
    
    probabilities_list = []
    for sample_index in range(len(name_index)):
        means=torch.tensor(name_index[sample_index],dtype=torch.float32,device=device)
        distributions=[Normal(mean,sigma) for mean in means]
        values=torch.arange(sequence_length,dtype=torch.float32,device=device)
        summed_probabilities=torch.zeros(sequence_length)
        for distribution in distributions:
            summed_probabilities+=torch.exp(distribution.log_prob(values))
        summed_probabilities /= summed_probabilities.sum() + epsilon

        probabilities_list.append(summed_probabilities)
    name_probability = torch.stack(probabilities_list)

    return [fact_tensor[0],name_probability,charge_labels]

dataloader = DataLoader(DataAdapterDataset('train'), batch_size=train_batch_size, shuffle=True, num_workers=0,
                        drop_last=False,collate_fn=collate_fn)

valid_dataloader = DataLoader(DataAdapterDataset('valid'), batch_size=test_batch_size, shuffle=False,num_workers=0,
                        drop_last=False,collate_fn=collate_fn)

test_dataloader = DataLoader(DataAdapterDataset('test'), batch_size = test_batch_size, shuffle=False,num_workers=0,
                        drop_last=False,collate_fn=collate_fn)


####模型构建
class MAMD(nn.Module):
    def __init__(self,embedding:np.array,input_dim=200,output_dim=charge_num,num_layers=2,dropout_rate=0.5,
                 bias=True,bidirectional=True,hidden_dim=256):
        super(MAMD,self).__init__()

        self.embs = nn.Embedding(164673, 200)
        self.embs.weight.data.copy_(embedding)
        self.embs.weight.requires_grad=False

        num_directions=2 if bidirectional else 1

        self.hidden_dim=hidden_dim

        self.rnn1=nn.GRU(input_size=input_dim,hidden_size=hidden_dim,num_layers=num_layers,bias=bias,
                         dropout=dropout_rate,bidirectional=bidirectional,batch_first=True)
        
        #全局注意力
        self.lin1=nn.Linear(in_features=hidden_dim*num_directions,out_features=hidden_dim*num_directions)
        self.Us1=nn.Parameter(torch.ones(hidden_dim*num_directions,1))

        #局部注意力
        self.lin2=nn.Linear(in_features=hidden_dim*num_directions,out_features=hidden_dim*num_directions)
        self.Us2=nn.Parameter(torch.ones(hidden_dim*num_directions,1))

        #聚合器
        self.beta=nn.Parameter(torch.ones(1))

        self.lin=nn.Linear(in_features=hidden_dim*num_directions,out_features=output_dim)

        self.tanh=nn.Tanh()



    def forward(self,fact:torch.Tensor,name_probability:torch.Tensor):
        fact_embedding=self.embs(fact)

        fact1,_=self.rnn1(fact_embedding)
        #第一项:[batch_size,seq_len,hidden_dim*num_directions]
        #第二项:[num_layers*num_directions,batch_size,hidden_dim*num_directions]
        u1=self.tanh(self.lin1(fact1))  #公式1
        a1=torch.matmul(u1,self.Us1)  #公式2
        alpha1=F.softmax(a1,dim=1).squeeze()  #公式3

        name_probability_diag=torch.zeros(name_probability.size(0),name_probability.size(1),
                                          name_probability.size(1)).to(device)
        # 将每行转换成对角矩阵
        for i in range(name_probability.size(0)):
            name_probability_diag[i]=torch.diag(name_probability[i])
        hc=torch.matmul(fact1.transpose(-1,-2),name_probability_diag)
        u2=self.tanh(self.lin2(hc.transpose(-1,-2)))
        a2=torch.matmul(u2,self.Us2)
        alpha2=F.softmax(a2,dim=1).squeeze()  #公式3
            
        am=alpha1*self.beta+alpha2  #公式4
        alpham=F.softmax(am,dim=1).squeeze()  #公式5

        g=torch.matmul(hc,alpham.unsqueeze(2)).squeeze()  #公式6
        output=self.lin(g)
        

        return output


model=MAMD(w)
model=model.to(device)



criterion=nn.CrossEntropyLoss()
optimizer=optim.RMSprop(model.parameters(),lr=learning_rate)



#训练
max_accuracy=0
now_patience=0
for epoch in range(max_epoch):
    #训练
    for step,batch in enumerate(tqdm(dataloader)):
        model.train()
        optimizer.zero_grad()

        facts_embedding=batch[0].to(device)
        names=batch[1].to(device)
        labels=batch[2].to(device)

        o=model(facts_embedding,names)

        loss_charge=criterion(o,labels)
        
        wandb.log({"loss":loss_charge})
        loss_charge.backward()
        optimizer.step()
    
    #验证
    with torch.no_grad():
        valid_label=[]
        valid_prediction=[]
        for step,batch in enumerate(tqdm(valid_dataloader)):
            model.eval()

            facts_embedding=batch[0].to(device)
            names=batch[1].to(device)
            labels=batch[2].tolist()

            valid_label.extend(labels)

            o=model(facts_embedding,names)
            predictions=o.cpu().argmax(dim=1).tolist()
            valid_prediction.extend(predictions)

    this_accuracy=accuracy_score(valid_label,valid_prediction)
    wandb.log({"epoch":epoch,"val_acc":this_accuracy})
    if this_accuracy>max_accuracy:
        max_accuracy=this_accuracy
        now_patience=0
    else:
        now_patience+=1
        if now_patience>patience:
            break

#测试
valid_label=[]
valid_prediction=[]
with torch.no_grad():
    for step,batch in enumerate(tqdm(test_dataloader)):
        model.eval()

        facts_embedding=batch[0].to(device)
        names=batch[1].to(device)
        labels=batch[2].tolist()

        valid_label.extend(labels)

        o=model(facts_embedding,names)
        predictions=o.cpu().argmax(dim=1).tolist()
        valid_prediction.extend(predictions)

print(accuracy_score(valid_label,valid_prediction))
print(precision_score(valid_label,valid_prediction,average='macro'))
print(recall_score(valid_label,valid_prediction,average='macro'))
print(f1_score(valid_label,valid_prediction,average='macro'))
wandb.log({"acc":accuracy_score(valid_label,valid_prediction),
           "p":precision_score(valid_label,valid_prediction,average='macro'),
           "r":recall_score(valid_label,valid_prediction,average='macro'),
           "f1":f1_score(valid_label,valid_prediction,average='macro')})

wandb.finish()