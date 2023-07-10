import json,argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

parser=argparse.ArgumentParser()
parser.add_argument('--small_law')
parser.add_argument('--small_charge')
parser.add_argument('--big_law')
parser.add_argument('--big_charge')
args=parser.parse_args()

law=json.load(open('law.json'))
charge=json.load(open('charge_details.json'))

word2vec=json.load(open('word2vec/word2vec.json'))
for k in word2vec:
    word2vec[k]=[float(factor) for factor in word2vec[k]]
word2vec['UNK']=np.mean(list(word2vec.values()),axis=0).tolist()

id2law={}
art_matrix=np.ndarray((103,200))
with open(args.small_law) as f:
    f_list=f.readlines()
for i in range(len(f_list)):
    law_order=f_list[i].strip()
    id2law[str(i)]=law_order
    art_matrix[i,:]=np.mean([word2vec[word] if word in word2vec else word2vec['UNK'] for word in law[law_order].split()],0)
json.dump(id2law,open('small_files/id2article.json','w'))

similarity=cosine_similarity(art_matrix)
art_tong={}
for i in range(103):
    art_tong[f_list[i].strip()]=[]
    for j in range(103):
        if i==j:
            continue
        if similarity[i,j]>0.95:
            art_tong[f_list[i].strip()].append(int(f_list[j].strip()))
json.dump(art_tong,open('small_files/art_tong.json','w'))

id2charge={}
charge_matrix=np.ndarray((119,200))
with open(args.small_charge) as f:
    f_list=f.readlines()
for i in range(len(f_list)):
    charge_name=f_list[i].strip()
    id2charge[str(i)]=charge_name
    charge_matrix[i,:]=np.mean([word2vec[word] if word in word2vec else word2vec['UNK'] for word in charge[charge_name]['定义'].split()],0)
json.dump(id2charge,open('small_files/id2charge.json','w'),ensure_ascii=False)

similarity=cosine_similarity(charge_matrix)
charge_tong={}
for i in range(119):
    charge_tong[f_list[i].strip()]=[]
    for j in range(119):
        if i==j:
            continue
        if similarity[i,j]>0.85:
            charge_tong[f_list[i].strip()].append(f_list[j].strip())
json.dump(charge_tong,open('small_files/charge_tong.json','w'),ensure_ascii=False)


id2law={}
art_matrix=np.ndarray((118,200))
with open(args.big_law) as f:
    f_list=f.readlines()
for i in range(len(f_list)):
    law_order=f_list[i].strip()
    id2law[str(i)]=law_order
    art_matrix[i,:]=np.mean([word2vec[word] if word in word2vec else word2vec['UNK'] for word in law[law_order].split()],0)
json.dump(id2law,open('big_files/id2article.json','w'))

similarity=cosine_similarity(art_matrix)
art_tong={}
for i in range(118):
    art_tong[f_list[i].strip()]=[]
    for j in range(118):
        if i==j:
            continue
        if similarity[i,j]>0.95:
            art_tong[f_list[i].strip()].append(int(f_list[j].strip()))
json.dump(art_tong,open('big_files/art_tong.json','w'))

id2charge={}
charge_matrix=np.ndarray((130,200))
with open(args.big_charge) as f:
    f_list=f.readlines()
for i in range(len(f_list)):
    charge_name=f_list[i].strip()
    id2charge[str(i)]=charge_name
    charge_matrix[i,:]=np.mean([word2vec[word] if word in word2vec else word2vec['UNK'] for word in charge[charge_name]['定义'].split()],0)
json.dump(id2charge,open('big_files/id2charge.json','w'),ensure_ascii=False)

similarity=cosine_similarity(charge_matrix)
charge_tong={}
for i in range(130):
    charge_tong[f_list[i].strip()]=[]
    for j in range(130):
        if i==j:
            continue
        if similarity[i,j]>0.85:
            charge_tong[f_list[i].strip()].append(f_list[j].strip())
json.dump(charge_tong,open('big_files/charge_tong.json','w'),ensure_ascii=False)