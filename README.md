本项目专注于复现CAIL2018数据集上的LJP工作。  
我专门写了篇综述，等写完了我挂到ArXiv上。  
项目代码的逻辑顺序是：数据集的不同预处理方法→不同的解决方案

超过5M的文件都存储在了百度网盘上，以方便大陆用户下载。

* [1. 数据](#数据)
* [2. LJP工作](#LJP工作)
* [3. 通用文本分类](#通用文本分类)

# 数据
CAIL2018数据集，原始数据来自裁判文书网，经处理后输入是事实描述文本，输出是案件的罪名、刑期、适用法条和罚金。  
中国大陆刑事一审案件，分成big和small两个子数据集。  

数据集处理策略：
## 原生数据格式
下载地址：<https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip>

以事实文本作为输入，以分类任务的范式，预测罪名（accusation）、法条（law）、刑期（imprisonment，单位为月，如被判为无期徒刑则是-1、死刑是-2

训练集是first_stage/train.json，测试集是 first_stage/test.json + restData/rest_data.json（文中说，这个配置是删除多被告情况，仅保留单一被告的案例；删除了出现频数低于30的罪名和法条；删除了不与特定罪名相关的102个法条（没看懂这句话是啥意思））

具体的待补
## LADAN格式
small数据集：  
链接：<https://pan.baidu.com/s/1kLueQRCFYYnYCOK9DE8o9Q>  
提取码：n51y

big数据集：  
链接：<https://pan.baidu.com/s/1EY-NowCigua0XQ5pwqenow>  
提取码：mkos 

具体的创建过程我没记录，总之是跟LADAN官方代码和统计信息是一样的，大概来说应该是看LADAN的GitHub项目得到的。

|**Dataset**|**small**|**big**|
|--|--|--|
|train cases|101,619|1,587,979|
|valid cases|13,768|-|
|test cases|26,749|185,120|
|articles|103|118|
|charges|119|130|
|term of penalty|11|11|
## CTM格式
待补
# LJP工作
论文前面的单选框表示是否完成并上传复现代码。代码具体复现了多少看models文件夹里面。  
因为我感觉不同数据集之间的转换不难，所以我就只复现一种数据集格式了（一般用的是LADAN格式）

**2023年**  
1. [ ] (ICAIL) [Legal Syllogism Prompting: Teaching Large Language Models for Legal Judgment Prediction](https://arxiv.org/abs/2307.08321)：法律三段论提示工程
2. [ ] (SIGIR) [ML-LJP: Multi-Law Aware Legal Judgment Prediction](https://dl.acm.org/doi/10.1145/3539618.3591731)：一个是将LJP任务扩展到多标签场景下（multi-law），一个是用GAT学习法律条文之间的相互作用以预测刑期，一个是对数字进行表征
3. [ ] (TOIS) [Contrastive Learning for Legal Judgment Prediction](https://dl.acm.org/doi/abs/10.1145/3580489)：“相似”样本是（1）法律的同一章节中的各种法律条文 （2）同一法律条文或相关法律条文的类似指控，即具有相同文章/指控标签的案件
2. [ ] (IPM) [LA-MGFM: A legal judgment prediction method via sememe-enhanced graph neural networks and multi-graph fusion mechanism](https://www.sciencedirect.com/science/article/pii/S0306457323001929)：构建语义图并融合
5. [ ] (Journal of King Saud University - Computer and Information Sciences) [TaSbeeb: A judicial decision support system based on deep learning framework](https://www.sciencedirect.com/science/article/pii/S1319157823002495)：沙特法院，所以这篇是伊斯兰教律法体系，特点是需要检索《古兰经》和圣训的经文。作者还在论文里吐槽阿拉伯文的数据集太少了，什么时候像CAIL一样搞个比赛就好了

**2022年**  
1. [ ] (AAAI) [Re6：读论文 LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification fro](https://blog.csdn.net/PolarisRisingWar/article/details/125192379)  
2. [ ] (AAAI) [Re14：读论文 ILLSI Interpretable Low-Resource Legal Decision Making](https://blog.csdn.net/PolarisRisingWar/article/details/126033696)
2. [ ] (ACL) [Re11：读论文 EPM Legal Judgment Prediction via Event Extraction with Constraints](https://blog.csdn.net/PolarisRisingWar/article/details/126029464)
3. [x] (IJCAI) [Re28：读论文 CECP Charge Prediction by Constitutive Elements Matching of Crimes](https://blog.csdn.net/PolarisRisingWar/article/details/126484229)
4. [ ] (IPM) [Re36：读论文 CEEN Improving legal judgment prediction through reinforced criminal element extraction](https://blog.csdn.net/PolarisRisingWar/article/details/127557195)
5. [x] (COLING) [Re 39：读论文 CTM Augmenting Legal Judgment Prediction with Contrastive Case Relations](https://blog.csdn.net/PolarisRisingWar/article/details/127515132)
6. [ ] (Artificial Intelligence and Law) [Re41：NumLJP Judicial knowledge‑enhanced magnitude‑aware reasoning for numerical legal judgment predi](https://link.springer.com/article/10.1007/s10506-022-09337-4)

**2021年**  
1. [ ] (ACL) [Re16：读论文 ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation](https://blog.csdn.net/PolarisRisingWar/article/details/126037188)
2. [ ] (NAACL) [Re18：读论文 GCI Everything Has a Cause: Leveraging Causal Inference in Legal Text Analysis](https://blog.csdn.net/PolarisRisingWar/article/details/126038513)
3. [ ] (SIGIR) [Re21：读论文 MSJudge Legal Judgment Prediction with Multi-Stage Case Representation Learning in the Real](https://blog.csdn.net/PolarisRisingWar/article/details/126054985)
4. [x] (SIGIR) [Re38：读论文 NeurJudge: A Circumstance-aware Neural Framework for Legal Judgment Prediction](https://blog.csdn.net/PolarisRisingWar/article/details/128243315)


**2020年**  
1. [x] (ACL) [Re27：读论文 LADAN Distinguish Confusing Law Articles for Legal Judgment Prediction](https://blog.csdn.net/PolarisRisingWar/article/details/126472752)

**2019年**
1. [ ] (EMNLP) [Charge-Based Prison Term Prediction with Deep Gating Network](https://aclanthology.org/D19-1667/)
2. [ ] (IJCAI) [Legal Judgment Prediction via Multi-Perspective Bi-Feedback Network](https://arxiv.org/abs/1905.03969)
3. [ ] (Law in Context) [A Brief History of the Changing Roles of Case Prediction in AI and Law](https://journals.latrobe.edu.au/index.php/law-in-context/article/view/88)：主要是美国那边LJP工作的综述

**2018年**
1. [ ] (EMNLP) [Legal Judgment Prediction via Topological Learning](https://aclanthology.org/D18-1390/)

**2017年**  
1. [ ] (EMNLP) [Re7：读论文 FLA/MLAC/FactLaw Learning to Predict Charges for Criminal Cases with Legal Basis](https://blog.csdn.net/PolarisRisingWar/article/details/125957914)

# 通用文本分类
1. [x] Bi-GRU
2. [x] SVM

**2018年**
1. [x] (NAACL) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)
# 结果展示
待补
## 原始格式数据集

## LADAN格式数据集