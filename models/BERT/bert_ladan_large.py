#基于LADAN预处理后的large数据集运行Bert文本分类

import argparse,os
import pickle as pk
from datetime import datetime

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel,AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--bert_folder')  #Bert预训练模型权重存储位置
parser.add_argument('--data_folder')  #数据集文件夹
parser.add_argument('--checkpoint_folder')  #checkpoint保存到的路径
args = parser.parse_args()

gpu_device='cuda:0'
epoch_num=8
train_batch_size=16
inference_batch_size=192

tokenizer=AutoTokenizer.from_pretrained(args.bert_folder)

print(datetime.now())




#数据集
class TextInitializeDataset(Dataset):
    """案例事实文本和标签对"""
    def __init__(self,mode) -> None:
        data=pk.load(open(os.path.join(args.data_folder,mode+'_processed_thulac_Legal_basis.pkl'),'rb'))
        fact_list=data['fact']
        self.original_text=[x.replace(' ','') for x in fact_list]
        self.law_label=data['law_label_lists']
        self.charge_label=data['accu_label_lists']
        self.penalty_label=data['term_lists']
    
    def __getitem__(self, index):
        return [self.original_text[index],self.law_label[index],self.charge_label[index],self.penalty_label[index]]
    
    def __len__(self):
        return len(self.original_text)

def collate_fn(batch):
    pt_batch=tokenizer([b[0] for b in batch],padding=True,truncation=True,max_length=512,return_tensors='pt')
    law_labels=torch.tensor([b[1] for b in batch])
    charge_labels=torch.tensor([b[2] for b in batch])
    penalty_labels=torch.tensor([b[3] for b in batch])
    return {'law_labels':law_labels,'charge_labels':charge_labels,'penalty_labels':penalty_labels,
            'input_ids':pt_batch['input_ids'],'token_type_ids':pt_batch['token_type_ids'],'attention_mask':pt_batch['attention_mask']}

train_data=TextInitializeDataset(mode='train')
train_dataloader=DataLoader(train_data,batch_size=train_batch_size,shuffle=True,collate_fn=collate_fn)
test_data=TextInitializeDataset(mode='test')
test_dataloader=DataLoader(test_data,batch_size=inference_batch_size,shuffle=False,collate_fn=collate_fn)



#建模
class BertClsModel(nn.Module):
    def __init__(self):
        super(BertClsModel,self).__init__()

        self.encoder=AutoModel.from_pretrained(args.bert_folder)
        self.dropout=nn.Dropout(0.1)
        self.law_classifier=nn.Linear(768,118)
        self.charge_classifier=nn.Linear(768,130)
        self.penalty_classifier=nn.Linear(768,11)
    
    def forward(self,input_ids,token_type_ids,attention_mask):
        x=self.encoder(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)['pooler_output']
        x=self.dropout(x)
        law_x=self.law_classifier(x)
        charge_x=self.charge_classifier(x)
        penalty_x=self.penalty_classifier(x)

        return (law_x,charge_x,penalty_x)



#训练
model=BertClsModel()
model.to(gpu_device)

optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-5)
loss_func=torch.nn.CrossEntropyLoss()

#训练
model.train()
for epoch in range(epoch_num):
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs=model(input_ids=batch['input_ids'].to(gpu_device),token_type_ids=batch['token_type_ids'].to(gpu_device),
                        attention_mask=batch['attention_mask'].to(gpu_device))
        train_loss=loss_func(outputs[0],batch['law_labels'].to(gpu_device))+loss_func(outputs[1],batch['charge_labels'].to(gpu_device))+\
                    loss_func(outputs[2],batch['penalty_labels'].to(gpu_device))
        train_loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(),os.path.join(args.checkpoint_folder,'bert_epoch'+str(epoch)+'.pth'))
    print('epoch'+str(epoch)+'训练完毕！当前时间是：'+str(datetime.now()))


#测试
with torch.no_grad():
    model.eval()
    law_label=[]
    law_predict=[]
    charge_label=[]
    charge_predict=[]
    penalty_label=[]
    penalty_predict=[]
    for batch in test_dataloader:
        outputs=model(input_ids=batch['input_ids'].to(gpu_device),token_type_ids=batch['token_type_ids'].to(gpu_device),
                    attention_mask=batch['attention_mask'].to(gpu_device))
        law_label.extend([i.item() for i in batch['law_labels']])
        law_predict.extend([i.item() for i in torch.argmax(outputs[0],1)])
        charge_label.extend([i.item() for i in batch['charge_labels']])
        charge_predict.extend([i.item() for i in torch.argmax(outputs[1],1)])
        penalty_label.extend([i.item() for i in batch['penalty_labels']])
        penalty_predict.extend([i.item() for i in torch.argmax(outputs[2],1)])
    print('法条：')
    print(accuracy_score(law_label,law_predict))
    print(precision_score(law_label,law_predict,average='macro'))
    print(recall_score(law_label,law_predict,average='macro'))
    print(f1_score(law_label,law_predict,average='macro'))
    print('罪名：')
    print(accuracy_score(charge_label,charge_predict))
    print(precision_score(charge_label,charge_predict,average='macro'))
    print(recall_score(charge_label,charge_predict,average='macro'))
    print(f1_score(charge_label,charge_predict,average='macro'))
    print('刑期：')
    print(accuracy_score(penalty_label,penalty_predict))
    print(precision_score(penalty_label,penalty_predict,average='macro'))
    print(recall_score(penalty_label,penalty_predict,average='macro'))
    print(f1_score(penalty_label,penalty_predict,average='macro'))

print('代码运行结束！当前时间是：'+str(datetime.now()))