不复现非顶会/顶刊且非经典的工作，但也列举在这里

超过5M的文件都存储在了百度网盘上，以方便大陆用户下载：  
链接：  
提取码：

* [1. 数据](#数据)
* [2. LJP工作](#LJP工作)
* [3. 通用文本分类](#通用文本分类)

# 数据
简单介绍：
|**数据集名称**|**国籍属性**|**下载和预处理策略**|**出处**|**任务形式**|
|---|---|---|-----|---|
|CAIL2018|中国大陆刑法|1. 原生数据格式<br>2. LADAN策略<br>3. CTM策略|(2018) [CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction](https://arxiv.org/abs/1807.02478)|输入事实描述文本，预测案件的罪名、刑期（分类任务或者ordinal分类任务）、适用法条|
|AIJudge|中国大陆|<https://www.datafountain.cn/competitions/277>|||
|ILSI|印度（英文）||(2022 AAAI) [Re6：读论文 LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification fro](https://blog.csdn.net/PolarisRisingWar/article/details/125192379)|根据事实描述文本和案件引用图，预测案件的适用法条|

其他相关数据集介绍：  
[LegalAI公开数据集的整理、总结及介绍（持续更新ing…）](https://blog.csdn.net/PolarisRisingWar/article/details/126058246)

TODO：
- [ ] 数据预处理方案和中间数据

# LJP工作
论文前面打钩的就是已经复现了（代码也放上来了），放了个空选择框的就是准备复现。没有选择框的就是我不准备复现了。

**2023年**  
1. [ ] (ICAIL) [Legal Syllogism Prompting: Teaching Large Language Models for Legal Judgment Prediction](https://arxiv.org/abs/2307.08321)：法律三段论提示工程
2. [ ] (SIGIR) [ML-LJP: Multi-Law Aware Legal Judgment Prediction](https://dl.acm.org/doi/10.1145/3539618.3591731)：一个是将LJP任务扩展到多标签场景下（multi-law），一个是用GAT学习法律条文之间的相互作用以预测刑期，一个是对数字进行表征
3. [ ] (TOIS) [Contrastive Learning for Legal Judgment Prediction](https://dl.acm.org/doi/abs/10.1145/3580489)：“相似”样本是（1）法律的同一章节中的各种法律条文 （2）同一法律条文或相关法律条文的类似指控，即具有相同文章/指控标签的案件
2. (IPM) [LA-MGFM: A legal judgment prediction method via sememe-enhanced graph neural networks and multi-graph fusion mechanism](https://www.sciencedirect.com/science/article/pii/S0306457323001929)：构建语义图并融合
5. (Journal of King Saud University - Computer and Information Sciences) [TaSbeeb: A judicial decision support system based on deep learning framework](https://www.sciencedirect.com/science/article/pii/S1319157823002495)：沙特法院，所以这篇是伊斯兰教律法体系，特点是需要检索《古兰经》和圣训的经文。作者还在论文里吐槽阿拉伯文的数据集太少了，什么时候像CAIL一样搞个比赛就好了

**2022年**  
1. [ ] (AAAI) [Re6：读论文 LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification fro](https://blog.csdn.net/PolarisRisingWar/article/details/125192379)  
2. [ ] (AAAI) [Re14：读论文 ILLSI Interpretable Low-Resource Legal Decision Making](https://blog.csdn.net/PolarisRisingWar/article/details/126033696)
2. [ ] (ACL) [Re11：读论文 EPM Legal Judgment Prediction via Event Extraction with Constraints](https://blog.csdn.net/PolarisRisingWar/article/details/126029464)
3. [x] (IJCAI) [Re28：读论文 CECP Charge Prediction by Constitutive Elements Matching of Crimes](https://blog.csdn.net/PolarisRisingWar/article/details/126484229)  
[代码：基于LADAN策略预处理的CAIL2018数据集上](models/CECP)
4. [ ] (IPM) [Re36：读论文 CEEN Improving legal judgment prediction through reinforced criminal element extraction](https://blog.csdn.net/PolarisRisingWar/article/details/127557195)
5. [x] (COLING) [Re 39：读论文 CTM Augmenting Legal Judgment Prediction with Contrastive Case Relations](https://blog.csdn.net/PolarisRisingWar/article/details/127515132)  
[代码：基于LADAN策略预处理的CAIL2018数据集上](models/CTM)
6. (Artificial Intelligence and Law) [Re41：NumLJP Judicial knowledge‑enhanced magnitude‑aware reasoning for numerical legal judgment predi](https://link.springer.com/article/10.1007/s10506-022-09337-4)

**2021年**  
1. [ ] (ACL) [Re16：读论文 ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation](https://blog.csdn.net/PolarisRisingWar/article/details/126037188)
2. [ ] (NAACL) [Re18：读论文 GCI Everything Has a Cause: Leveraging Causal Inference in Legal Text Analysis](https://blog.csdn.net/PolarisRisingWar/article/details/126038513)
3. [ ] (SIGIR) [Re21：读论文 MSJudge Legal Judgment Prediction with Multi-Stage Case Representation Learning in the Real](https://blog.csdn.net/PolarisRisingWar/article/details/126054985)
4. [x] (SIGIR) [Re38：读论文 NeurJudge: A Circumstance-aware Neural Framework for Legal Judgment Prediction](https://blog.csdn.net/PolarisRisingWar/article/details/128243315)  
[代码：基于LADAN策略预处理的CAIL2018数据集上](models/NeurJudge/CAIL2018_LADAN)


**2020年**  
1. [x] (ACL) [Re27：读论文 LADAN Distinguish Confusing Law Articles for Legal Judgment Prediction](https://blog.csdn.net/PolarisRisingWar/article/details/126472752)  
[代码：在基于LADAN策略预处理的CAIL2018数据集上](models/LADAN/CAIL2018_LADAN)

**2019年**
1. [ ] (EMNLP) [Charge-Based Prison Term Prediction with Deep Gating Network](https://aclanthology.org/D19-1667/)
2. [ ] (IJCAI) [Legal Judgment Prediction via Multi-Perspective Bi-Feedback Network](https://arxiv.org/abs/1905.03969)
3. (Law in Context) [A Brief History of the Changing Roles of Case Prediction in AI and Law](https://journals.latrobe.edu.au/index.php/law-in-context/article/view/88)：主要是美国那边LJP工作的综述

**2018年**
1. [ ] (EMNLP) [Legal Judgment Prediction via Topological Learning](https://aclanthology.org/D18-1390/)

**2017年**  
1. [ ] (EMNLP) [Re7：读论文 FLA/MLAC/FactLaw Learning to Predict Charges for Criminal Cases with Legal Basis](https://blog.csdn.net/PolarisRisingWar/article/details/125957914)

# 通用文本分类
1. [x] Bi-GRU  
[代码：在基于LADAN策略预处理的CAIL2018数据集上](models/BiGRU/CAIL2018_LADAN)
2. [x] SVM  
[代码：在基于LADAN策略预处理的CAIL2018数据集上](models/BiGRU/CAIL2018_LADAN)

**2018年**
1. [x] (NAACL) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)  
[代码：在基于LADAN策略预处理的CAIL2018数据集上](models/BERT/CAIL2018_LADAN)