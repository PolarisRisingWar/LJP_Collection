import argparse
import json
import logging
import os
import pickle as pk
import random

import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from model import NeurJudge
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

#超参设置
train_batch_size=64  #官方代码给的是64，我可以开到128
test_batch_size=256  #官方代码给的是256，我开到512会超，能开到256
train_epoch=16  #官方代码给的是16

###传入命令行参数
parser=argparse.ArgumentParser()
parser.add_argument('--data_folder')
parser.add_argument('--data_type')  #big/small
parser.add_argument('--gpu')
args=parser.parse_args()

###设置随机数和日志
random.seed(42)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

###导入word2vec矩阵
word2id=json.load(open('word2vec/word2id.json'))
word2vec=json.load(open('word2vec/word2vec.json'))
word2vec['UNK'] = np.random.randn(200).tolist() 
word2vec['Padding'] = [0. for i in range(200)]
embedding=np.zeros((339503,200))
for k in word2id:
    embedding[word2id[k],:]=[float(factor) for factor in word2vec[k]]
embedding=torch.from_numpy(embedding)

###设置cuda环境
os.environ['CUDA_VISIBLE_DEVICES'] =args.gpu
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

###设置辅助数据路径
charge_tong_file=os.path.join(args.data_type+'_files','charge_tong.json')
art_tong_file=os.path.join(args.data_type+'_files','art_tong.json')
id2charge_file=os.path.join(args.data_type+'_files','id2charge.json')
id2article_file=os.path.join(args.data_type+'_files','id2article.json')

###建模
if args.data_type=='small':
    charge_num=119
    law_num=103
else:
    charge_num=130
    law_num=118

model=NeurJudge(embedding,charge_tong_file,art_tong_file,id2charge_file,id2article_file,charge_num,law_num)
model=model.to(device)

###数据处理工具类
class Data_Process():
    def __init__(self):
        self.word2id = json.load(open('word2vec/word2id.json',"r"))
        self.symbol = [",", ".", "?", "\"", "”", "。", "？", "","，",",","、","”"]
        self.last_symbol = ["?", "。", "？"]
        self.charge2detail = json.load(open('charge_details.json','r'))
        self.sent_max_len = 200
        self.law = json.load(open('law.json'))

    def transform(self, word):
        if not (word in self.word2id.keys()):
            return self.word2id["UNK"]
        else:
            return self.word2id[word]
    
    def parse(self, sent):
        result = []
        sent = sent.strip().split()
        for word in sent:
            if len(word) == 0:
                continue
            if word in self.symbol:
                continue
            result.append(word)
        return result

    def seq2tensor(self, sents, max_len=350):
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, max_len)

        sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                sent_tensor[s_id][w_id] = self.transform(word) 
        return sent_tensor,sent_len    

    def get_graph(self):
        charge_tong = json.load(open(charge_tong_file))
        art_tong = json.load(open(art_tong_file))
        charge_tong2id = {}
        id2charge_tong = {}
        legals = []
        for index,c in enumerate(charge_tong):
            charge_tong2id[c] = str(index)
            id2charge_tong[str(index)] = c
        
        legals = []  
        for i in charge_tong:
            legals.append(self.parse(self.charge2detail[i]['定义']))
           
        legals,legals_len = self.seq2tensor(legals,max_len=100)

        art2id = {}
        id2art = {}
        for index,c in enumerate(art_tong):
            art2id[c] = str(index)
            id2art[str(index)] = c
        arts = []
        for i in art_tong:
            arts.append(self.parse(self.law[str(i)]))
        arts,arts_sent_lent = self.seq2tensor(arts,max_len=150)
        
        return legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art
        
    def process_data(self,data):
        fact_all = []
        for sentence in data[0]:
            fact_all.append(self.parse(sentence))

        article_label = data[1]
        charge_label = data[2]
        time_label = data[3]

        documents,sent_lent = self.seq2tensor(fact_all,max_len=350)
        return charge_label,article_label,time_label,documents,sent_lent
    
    def process_law(self,label_names,type = 'charge'):
        if type == 'charge':
            labels = []  
            for i in label_names:
                labels.append(self.parse(self.charge2detail[i]['定义']))
            labels , labels_len = self.seq2tensor(labels,max_len=100)
            return labels , labels_len
        else:
            labels = []  
            for i in label_names:
                labels.append(self.parse(self.law[str(i)]))
            labels , labels_len = self.seq2tensor(labels,max_len=150)
            return labels , labels_len


###这个是使用LADAN处理后格式数据用的
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



###训练部分
dataloader = DataLoader(DataAdapterDataset('train'), batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=False)
criterion = nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

process = Data_Process()
legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art = process.get_graph()

legals,legals_len,arts,arts_sent_lent = legals.to(device),legals_len.to(device),arts.to(device),arts_sent_lent.to(device)
num_epoch = train_epoch
global_step = 0
for epoch in range(num_epoch):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    logger.info("Trianing Epoch: {}/{}".format(epoch+1, int(num_epoch)))
    for step,batch in enumerate(tqdm(dataloader)):
        global_step += 1
        nb_tr_steps += 1
        model.train()
        optimizer.zero_grad()

        charge_label,article_label,time_label,documents,sent_lent = process.process_data(batch)
       
        documents = documents.to(device)
        charge_label = charge_label.to(device)
        article_label = article_label.to(device)
        time_label = time_label.to(device)
       
        sent_lent = sent_lent.to(device)
        charge_out,article_out,time_out = model(legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art,documents,sent_lent,process,device)
        
        loss_charge = criterion(charge_out,charge_label)
        loss_art = criterion(article_out,article_label)
        loss_time = criterion(time_out,time_label)

        loss = ( loss_charge + loss_art + loss_time )/3
        tr_loss+=loss.item()
        loss.backward()
        optimizer.step()

        if global_step%1000 == 0:
            logger.info("Training loss: {}, global step: {}".format(tr_loss/nb_tr_steps, global_step))




###测试部分
def get_value(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def gen_result(res):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_value(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_value(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    print("Micro precision\t%.4f" % micro_precision)
    print("Micro recall\t%.4f" % micro_recall)
    print("Micro f1\t%.4f" % micro_f1)
    print("Macro precision\t%.4f" % macro_precision)
    print("Macro recall\t%.4f" % macro_recall) 
    print("Macro f1\t%.4f" % macro_f1)

    return

def eval_data_types(target,prediction,num_labels):
    ground_truth_v2 = []
    predictions_v2 = []
    for i in target:
        v = [0 for _ in range(num_labels)]
        v[i] = 1
        ground_truth_v2.append(v)
    for i in prediction:
        v = [0 for _ in range(num_labels)]
        v[i] = 1
        predictions_v2.append(v)

    res = []
    for i in range(num_labels):
        res.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    y_true = np.array(ground_truth_v2)
    y_pred = np.array(predictions_v2)
    for i in range(num_labels):
    
        outputs1 = y_pred[:, i]
        labels1 = y_true[:, i] 
        res[i]["TP"] += int((labels1 * outputs1).sum())
        res[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        res[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        res[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())
    gen_result(res)

    return 0

id2charge = json.load(open(id2charge_file))
dataloader = DataLoader(DataAdapterDataset('test'), batch_size = test_batch_size, shuffle=False, num_workers=0, drop_last=False)
predictions_article = []
predictions_charge = []
predictions_time = []

true_article = []
true_charge = []
true_time = []


for step,batch in enumerate(tqdm(dataloader)):
    model.eval()
    charge_label,article_label,time_label,documents,sent_lent = process.process_data(batch)
    documents = documents.to(device)
    
    sent_lent = sent_lent.to(device)
    true_article.extend(article_label.numpy())
    true_charge.extend(charge_label.numpy())
    true_time.extend(time_label.numpy())

    with torch.no_grad():
        charge_out,article_out,time_out = model(legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art,documents,sent_lent,process,device)
        
    charge_pred = charge_out.cpu().argmax(dim=1).numpy()
    article_pred = article_out.cpu().argmax(dim=1).numpy()
    time_pred = time_out.cpu().argmax(dim=1).numpy()

    predictions_article.extend(article_pred)
    predictions_charge.extend(charge_pred)
    predictions_time.extend(time_pred)

print('罪名')
print(accuracy_score(true_charge,predictions_charge))
eval_data_types(true_charge,predictions_charge,num_labels=charge_num)
print('法条')
print(accuracy_score(true_article,predictions_article))
eval_data_types(true_article,predictions_article,num_labels=law_num)
print('刑期')
print(accuracy_score(true_time,predictions_time))
eval_data_types(true_time,predictions_time,num_labels=11)
