#big数据集上的数据获取

import pickle as pk
import random
from tqdm import tqdm

import thulac
model=thulac.thulac(seg_only=True)

import numpy as np

with open('/home/wanghuijuan/whj_files/github_projects/LADAN/data_and_config/data/w2id_thulac.pkl', 'rb') as f:
    word2id_dict = pk.load(f)

punctuations='。：（）；:“”，,'  #用这些标点符号作为分句指标，并且不保留这几个标点符号本身
max_sent_num=64
max_sent_len=32  #这两个是CECP原本的设置


random.seed(20230215)
dataset_path='/data/wanghuijuan/cail_ladan/legal_basis_data_big/'
with open(dataset_path+'train_processed_thulac_Legal_basis.pkl', 'rb') as f:
    R_train_total=pk.load(f)
sample_length=len(R_train_total['fact'])
print('原始训练集中含有'+str(len(R_train_total['fact']))+'个样本')

sample_index=list(range(sample_length))
random.shuffle(sample_index)
sample_index1=sample_index[:int(0.9*sample_length)]
R_train={key:[R_train_total[key][i] for i in sample_index1] for key in R_train_total.keys()}
print('最终使用的训练集中含有'+str(len(R_train['fact']))+'个样本')

sample_index2=sample_index[int(0.9*sample_length):]
R_valid={key:[R_train_total[key][i] for i in sample_index2] for key in R_train_total.keys()}
print('最终使用的验证集中含有'+str(len(R_valid['fact']))+'个样本')

with open(dataset_path+'test_processed_thulac_Legal_basis.pkl', 'rb') as f:
    R_test=pk.load(f)

data_map={'train':R_train,'valid':R_valid,'test':R_test}

for k in ['train','valid','test']:
    x=[]
    y=[]
    sent_num=[]
    sent_len=[]
    original_data=data_map[k]
    sample_num=len(original_data['fact'])
    print(sample_num)
    for i in tqdm(range(sample_num)):
        word_list=original_data['fact'][i].split()  #每个样本分词后的词语列表
        this_x=[]
        this_sent_len=[]
        current_sentence=[]
        for j in word_list:
            if j in punctuations:
                this_sent_len.append(len(current_sentence))
                if len(current_sentence)<max_sent_len:  #补全本句
                    for _ in range(max_sent_len-len(current_sentence)):
                        current_sentence.append(word2id_dict['BLANK'])
                this_x.append(current_sentence)
                if len(this_x)==max_sent_num:  #到达此样本最大句数
                    break
                current_sentence=[]
            else:
                if len(current_sentence)==max_sent_len:  #到达此句最大词数
                    continue
                if j in word2id_dict:
                    current_sentence.append(word2id_dict[j])
                else:
                    current_sentence.append(word2id_dict['UNK'])
        
        sent_num.append(len(this_x))

        if len(this_x)<max_sent_num:
            for _ in range(max_sent_num-len(this_x)):
                this_x.append([word2id_dict['BLANK'] for _ in range(max_sent_len)])  #补全本样本的所有句子
        
        x.append(this_x)
        y.append(original_data['accu_label_lists'][i])
        sent_len.append(this_sent_len)
    
    x=np.array(x)
    y=np.array(y)

    print(x.shape)
    print(y.shape)
    print(len(sent_num))
    print(len(sent_len))  #list不变

    
    to_path='/data/wanghuijuan/cecp_data/cail_big_'+k+'.pkl'
    with open(to_path, 'wb') as f:
        pk.dump({'x':x, 'y':y,'sent_num':sent_num,'sent_len':sent_len}, f)







