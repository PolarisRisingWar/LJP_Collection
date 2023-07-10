先立个flag放在这里。我现在项目不是垂直LJP方向了所以不再chase这个方向了……反正flag就放在这里了。

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
|CAIL2018|中国大陆刑法|1. 原生数据格式<br>2. LADAN策略<br>3. CTM策略||输入事实描述文本，预测案件的罪名、刑期（分类任务或者ordinal分类任务）、适用法条|
|ILSI|印度（英文）||[Re6：读论文 LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification fro](https://blog.csdn.net/PolarisRisingWar/article/details/125192379)|根据事实描述文本和案件引用图，预测案件的适用法条|

其他相关数据集介绍：  
[LegalAI公开数据集的整理、总结及介绍（持续更新ing…）](https://blog.csdn.net/PolarisRisingWar/article/details/126058246)

# LJP工作
**2022年**  
1. [ ] (AAAI) [Re6：读论文 LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification fro](https://blog.csdn.net/PolarisRisingWar/article/details/125192379)  
2. [ ] (AAAI) [Re14：读论文 ILLSI Interpretable Low-Resource Legal Decision Making](https://blog.csdn.net/PolarisRisingWar/article/details/126033696)
2. [ ] (ACL) [Re11：读论文 EPM Legal Judgment Prediction via Event Extraction with Constraints](https://blog.csdn.net/PolarisRisingWar/article/details/126029464)
3. [ ] (IJCAI) [Re28：读论文 CECP Charge Prediction by Constitutive Elements Matching of Crimes](https://blog.csdn.net/PolarisRisingWar/article/details/126484229)
4. [ ] (IPM) [Re36：读论文 CEEN Improving legal judgment prediction through reinforced criminal element extraction](https://blog.csdn.net/PolarisRisingWar/article/details/127557195)
5. [ ] (COLING) [Re 39：读论文 CTM Augmenting Legal Judgment Prediction with Contrastive Case Relations](https://blog.csdn.net/PolarisRisingWar/article/details/127515132)

**2021年**  
1. [ ] (ACL) [Re16：读论文 ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation](https://blog.csdn.net/PolarisRisingWar/article/details/126037188)
2. [ ] (NAACL) [Re18：读论文 GCI Everything Has a Cause: Leveraging Causal Inference in Legal Text Analysis](https://blog.csdn.net/PolarisRisingWar/article/details/126038513)
3. [ ] (SIGIR) [Re21：读论文 MSJudge Legal Judgment Prediction with Multi-Stage Case Representation Learning in the Real](https://blog.csdn.net/PolarisRisingWar/article/details/126054985)
4. [x] (SIGIR) [Re38：读论文 NeurJudge: A Circumstance-aware Neural Framework for Legal Judgment Prediction](https://blog.csdn.net/PolarisRisingWar/article/details/128243315)
[NeurJudge在基于LADAN策略预处理的CAIL2018数据集上实现LJP任务](models/NeurJudge/CAIL2018_LADAN)


**2020年**  
1. [x] (ACL) [Re27：读论文 LADAN Distinguish Confusing Law Articles for Legal Judgment Prediction](https://blog.csdn.net/PolarisRisingWar/article/details/126472752)
[LADAN在基于LADAN策略预处理的CAIL2018数据集上实现LJP任务](models/LADAN/CAIL2018_LADAN)

**2019年**
1. [ ] (EMNLP) [Charge-Based Prison Term Prediction with Deep Gating Network](https://aclanthology.org/D19-1667/)
2. [ ] (IJCAI) [Legal Judgment Prediction via Multi-Perspective Bi-Feedback Network](https://arxiv.org/abs/1905.03969)

**2018年**
1. [ ] (EMNLP) [Legal Judgment Prediction via Topological Learning](https://aclanthology.org/D18-1390/)

**2017年**  
1. [ ] (EMNLP) [Re7：读论文 FLA/MLAC/FactLaw Learning to Predict Charges for Criminal Cases with Legal Basis](https://blog.csdn.net/PolarisRisingWar/article/details/125957914)

# 通用文本分类
1. [x] Bi-GRU
[Bi-GRU在基于LADAN策略预处理的CAIL2018数据集上实现文本分类模型](models/BiGRU/CAIL2018_LADAN)
2. [x] SVM
[SVM在基于LADAN策略预处理的CAIL2018数据集上实现文本分类模型](models/BiGRU/CAIL2018_LADAN)

**2018年**
1. [x] (NAACL) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)
[BERT在基于LADAN策略预处理的CAIL2018数据集上实现文本分类模型](models/BERT/CAIL2018_LADAN)