import wandb
import pickle as pk
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


device="cuda:3"

#原论文或项目文件中的超参：
hidden_size=256
max_one_sentence_word_num=128
max_one_document_sentence_num=32
learning_rate=1e-3
weight_decay=1e-3
dropout_rate=0.5
batch_size=128
epoch_num=16
filters=64
min_gram=2
max_gram=5
fc1_feature=256

#dataset-specific超参：
law_num=
charge_num=
penalty_num=
embedding_dim=200

wandb.init(
    project="",
    
    config={
    "model":"TopJudge"
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
    for sentence in case.split("。"):
        if not sentence=="":
            sentence_words=[]
            sentence=sentence.strip().split()
            for word in sentence:
                if len(word) == 0:
                    continue
                if word in [",", ".", "?", "\"", "”", "。", "？", "","，",",","、","”"]:
                    continue
                sentence_words.append(transform_word2id(word))
            result.append(sentence_words)
    return result

def word_list2tensor(word_list:list[list]):
    """将一个batch转换为一个张量"""
    sentence_num=min(max(len(case) for case in word_list),max_one_document_sentence_num)
    word_num_in_one_sentence=min(max([len(sentence) for sentence in [case for case in word_list]]),
                                 max_one_sentence_word_num)
    # 初始化一个全0的张量
    tensor = torch.zeros(len(word_list),sentence_num,word_num_in_one_sentence,dtype=torch.long)

    # 填充张量
    for doc_idx, doc in enumerate(word_list):
        for sent_idx, sent in enumerate(doc):
            if sent_idx < sentence_num:  # 检查句子索引是否小于最大句子数
                for word_idx, word in enumerate(sent):
                    if word_idx < word_num_in_one_sentence:  # 检查词索引是否小于最大词数
                        tensor[doc_idx, sent_idx, word_idx] = word

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
    """https://github.com/thunlp/TopJudge/blob/c9186b132e79830fd4e855777b06a601d76bf0a2/net/model/encoder/cnn_encoder.py"""
    def __init__(self):
        super(CNNEncoder,self).__init__()

        self.convs=[]
        for a in range(min_gram,max_gram+1):
            self.convs.append(nn.Conv2d(1,filters,(a,embedding_dim)))

        self.convs=nn.ModuleList(self.convs)

    def forward(self,x):
        sample_num=x.size()[0]
        sentence_num=x.size()[1]
        sentence_len=x.size()[2]
        x=x.view(sample_num,1,-1,embedding_dim)
        conv_out=[]
        gram=min_gram
        for conv in self.convs:
            y=F.relu(conv(x)).view(sample_num,filters,-1)
            y=F.max_pool1d(y,kernel_size=sentence_num*sentence_len-gram+1).view(sample_num,-1)
            conv_out.append(y)
            gram+=1

        conv_out=torch.cat(conv_out,dim=1)

        fc_input=conv_out

        features=(max_gram-min_gram+1) *filters

        fc_input = fc_input.view(-1, features)

        return fc_input



def generate_graph():
    s = "[(1 2),(2 3),(1 3)]"
    arr = s.replace("[", "").replace("]", "").split(",")
    graph = []
    n = 0
    if (s == "[]"):
        arr = []
        n = 3
    for a in range(0, len(arr)):
        arr[a] = arr[a].replace("(", "").replace(")", "").split(" ")
        arr[a][0] = int(arr[a][0])
        arr[a][1] = int(arr[a][1])
        n = max(n, max(arr[a][0], arr[a][1]))

    n += 1
    for a in range(0, n):
        graph.append([])
        for _ in range(0, n):
            graph[a].append(False)

    for a in range(0, len(arr)):
        graph[arr[a][0]][arr[a][1]] = True

    return graph



class LSTMDecoder(nn.Module):
    def __init__(self):
        super(LSTMDecoder,self).__init__()
        self.feature_len=hidden_size

        features=hidden_size
        self.hidden_dim=features
        
        self.outfc=[nn.Linear(features,law_num),nn.Linear(features,charge_num),nn.Linear(features,penalty_num)]

        self.midfc = []
        for _ in range(3):
            self.midfc.append(nn.Linear(features, features))

        self.cell_list=[None]
        for _ in range(3):
            self.cell_list.append(nn.LSTMCell(hidden_size,hidden_size))

        self.hidden_state_fc_list=[]
        for _ in range(4):
            arr = []
            for _ in range(4):
                arr.append(nn.Linear(features, features))
            arr=nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list=[]
        for _ in range(4):
            arr = []
            for _ in range(4):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self):
        self.hidden_list = []
        
        for _ in range(4):
            if torch.cuda.is_available():
                self.hidden_list.append((
                    torch.autograd.Variable(
                        torch.zeros(batch_size, self.hidden_dim).to(device)),
                    torch.autograd.Variable(
                        torch.zeros(batch_size, self.hidden_dim).to(device))))
            else:
                self.hidden_list.append((
                    torch.autograd.Variable(torch.zeros(batch_size, self.hidden_dim)),
                    torch.autograd.Variable(torch.zeros(batch_size, self.hidden_dim))))

    def forward(self,x):
        fc_input = x
        outputs = []

        sample_num=x.size()[0]

        graph = generate_graph()

        first = []
        for a in range(4):
            first.append(True)
        for a in range(1, 4):
            hx=self.hidden_list[a][0][:sample_num]
            cx=self.hidden_list[a][1][:sample_num]
            h, c = self.cell_list[a](fc_input, (hx,cx))
            for b in range(1, 4):
                if graph[a][b]:
                    hp, cp = self.hidden_list[b]
                    if first[b]:
                        first[b] = False
                        hp, cp = h, c
                    else:
                        hp = hp + self.hidden_state_fc_list[a][b](h)
                        cp = cp + self.cell_state_fc_list[a][b](c)
                    self.hidden_list[b] = (hp, cp)

            outputs.append(self.outfc[a - 1](h).view(sample_num, -1))

        return outputs


class TopJudge(nn.Module):
    """https://github.com/thunlp/TopJudge/blob/c9186b132e79830fd4e855777b06a601d76bf0a2/net/model/model/cnn_seq.py"""
    def __init__(self,embedding:np.array,dropout_rate:float=dropout_rate):
        super(TopJudge,self).__init__()

        self.embs = nn.Embedding(164673, 200)
        self.embs.weight.data.copy_(embedding)
        self.embs.weight.requires_grad=False

        self.encoder=CNNEncoder()
        self.decoder = LSTMDecoder()
        self.dropout = nn.Dropout(dropout_rate)

    def init_hidden(self, ):
        self.decoder.init_hidden()

    def forward(self,x):
        x=self.embs(x)
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.decoder(x)

        return x


model=TopJudge(w)
model=model.to(device)



criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)



#训练#https://github.com/thunlp/TopJudge/blob/c9186b132e79830fd4e855777b06a601d76bf0a2/net/work.py
max_accuracy=0
now_patience=0
for epoch in range(epoch_num):
    #训练
    for step,batch in enumerate(tqdm(dataloader)):
        model.train()
        optimizer.zero_grad()

        facts_embedding=batch[0].to(device)
        law_labels=batch[1].to(device)
        charge_labels=batch[2].to(device)
        penalty_labels=batch[3].to(device)

        model.init_hidden()
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

        model.init_hidden()
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