#预处理代码
#elements文件

import pickle as pk
import numpy as np

import thulac
model=thulac.thulac(seg_only=True)

#标签数字到文本的对应见这个：
id2charge=open('/data/wanghuijuan/cail_ladan/LADAN-processed-big/new_accu.txt').readlines()

#然后CE文本是这两个：
CE1=open('whj_files/github_projects/CECP/data/CEs/CEs.txt',encoding='gbk').readlines()
CE2=open('whj_files/github_projects/CECP/data/CEs/CEs_supp.txt').readlines()
CE=CE1+CE2


charge_num=len(id2charge)
print(charge_num)

with open('/home/wanghuijuan/whj_files/github_projects/LADAN/data_and_config/data/w2id_thulac.pkl', 'rb') as f:
    word2id_dict = pk.load(f)

def get_num(text:str):
    """输入未分词的文本，分词（LADAN官方预处理代码是不去除停用词的）→转换为数值（tab或者cut），返回数值列表"""
    word_list=[a for a in model.cut(text,text=True).split(' ')]
    num_list=[]
    for word in word_list:
        if word in word2id_dict:
            num_list.append(word2id_dict[word])
        else:
            num_list.append(word2id_dict['UNK'])
    return num_list

def tab_num_list(num_list:list,length:int,blank_token:int=word2id_dict['BLANK']):
    """将指定列表截断或tab到指定长度"""
    if len(num_list)<length:
        num_list.extend([blank_token]*(length-len(num_list)))
    else:
        num_list=num_list[:length]
    return num_list


num2charge={}
ele_subject=[]
ele_subjective=[]
ele_object=[]
ele_objective=[]

for charge_id in range(charge_num):
    charge=id2charge[charge_id].strip()

    if charge=='走私普通货物、物品':
        charge='走私普通货物物品'

    for ce in CE:  #从CE集合中，找这个罪名对应的CE
        if ce.startswith(charge):
            #这就是那个CE

            #将CE拆为罪名和4个部分
            parts=[x.strip() for x in ce.split('&')]
            #5或6个元素。第一个元素是罪名，然后倒数4个分别是subject-subjective-object-objective

            num2charge[charge_id]=parts[0]
            ele_subject.append(tab_num_list(get_num(parts[-4]),100))
            ele_subjective.append(tab_num_list(get_num(parts[-3]),100))
            ele_object.append(tab_num_list(get_num(parts[-2]),200))
            ele_objective.append(tab_num_list(get_num(parts[-1]),400))


            break


to_path='/data/wanghuijuan/cecp_data/elements_cail_big.pkl'
with open(to_path, 'wb') as f:
    pk.dump({'num2charge':num2charge,'ele_subject':np.array(ele_subject),'ele_subjective':np.array(ele_subjective),
            'ele_object':np.array(ele_object),'ele_objective':np.array(ele_objective)}, f)