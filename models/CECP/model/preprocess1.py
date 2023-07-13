#预处理代码

import pickle as pk
import numpy as np

path='/data/wanghuijuan/cecp_data/'

path_train = path + 'criminal_small_train.pkl'
with open(path_train, 'rb') as f:
    data = pk.load(f)

print(type(data))
print(data.keys())
print(type(data['x']))
print(data['x'].shape)  #(61586, 64, 32)  样本数，一个文档的最长句数，一句的最长词数
print(data['y'].shape)  #(61586,)
print(type(data['sent_num']))
print(data['sent_num'][0])  #24
print(type(data['sent_len']))
print(data['sent_len'][0])  #[3, 17, 12, 22, 1, 12, 3, 4, 32, 24, 4, 17, 14, 3, 7, 7, 10, 24, 15, 4, 8, 4, 9, 2]
print(len(data['sent_len'][0]))  #24

path_elements = path + 'elements_criminal.pkl'
with open(path_elements, 'rb') as f:
    data = pk.load(f)

print(data.keys())
print(data['ele_subject'].shape)  #[罪名数,100]
print(data['ele_subjective'].shape)
print(data['ele_object'].shape)
print(data['ele_objective'].shape)
print(data['num2charge'].keys())  #这个捏就是ID与罪名文本的对应啦，啊我的问题就是这玩意对应起来有意义吗
print(data['num2charge'][0])

f_test = pk.load(open('/data/wanghuijuan/cail_ladan/legal_basis_data_small/test_processed_thulac_Legal_basis.pkl', 'rb'))
print(f_test.keys())



#sent_num和sent_len什么的我这边也按照CECP原文的来搞
with open('/home/wanghuijuan/whj_files/github_projects/LADAN/data_and_config/data/w2id_thulac.pkl', 'rb') as f:
    word2id_dict = pk.load(f)  #有'BLANK'和'UNK'的

punctuations='。：（）；:“”，,'  #用这些标点符号作为分句指标，并且不保留这几个标点符号本身
max_sent_num=64
max_sent_len=32

for k in ['train','valid','test']:
    x=[]
    y=[]
    sent_num=[]
    sent_len=[]
    path='/data/wanghuijuan/cail_ladan/legal_basis_data_small/'+k+'_processed_thulac_Legal_basis.pkl'
    with open(path, 'rb') as f:
        original_data=pk.load(f)
    sample_num=len(original_data['fact'])
    print(sample_num)
    for i in range(sample_num):
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

    
    to_path='/data/wanghuijuan/cecp_data/cail_small_'+k+'.pkl'
    with open(to_path, 'wb') as f:
        pk.dump({'x':x, 'y':y,'sent_num':sent_num,'sent_len':sent_len}, f)
    
        
