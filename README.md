本项目专注于复现CAIL2018数据集上的LJP工作。  
我专门写了篇综述，等写完了我挂到ArXiv上。  
项目代码的逻辑顺序是：数据集的不同预处理方法→不同的解决方案

超过5M的文件都存储在了百度网盘上，以方便大陆用户下载。

目录：
1. 数据
2. LJP paper list
3. 通用文本分类
4. 结果展示
5. 引用
6. star history

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
# LJP paper list
论文前面的单选框表示是否完成并上传复现代码。代码具体复现了多少看models文件夹里面。  
因为我感觉不同数据集之间的转换不难，所以我就只复现一种数据集格式了（一般用的是LADAN格式）**除了LADAN之外，我大多数代码直接使用了原文作为初始数据。fact_list是文本列表，每个元素是用空格隔开的分词后的原文**

**2024年**  
(ESWA) KnowPrompt4LJP [Chinese legal judgment prediction via knowledgeable prompt learning](https://www.sciencedirect.com/science/article/abs/pii/S0957417423026799) 我在另一个GitHub项目那边帮人下了这篇paper，有需自取：<https://github.com/PolarisRisingWar/pytorch_ljp/files/15269723/Chinese.legal.judgment.prediction.via.knowledgeable.prompt.learning.pdf>  
LegalDuet: Learning Effective Representations for Legal Judgment Prediction through a Dual-View Legal Clue Reasoning

**2023年**  
Precedent-Enhanced Legal Judgment Prediction with LLM and Domain-Model Collaboration  
A Comprehensive Evaluation of Large Language Models on Legal Judgment Prediction  
Exploiting Contrastive Learning and Numerical Evidence for Confusing Legal Judgment Prediction  
CL4LJP Contrastive Learning for Legal Judgment Prediction  
ML-LJP: Multi-Law Aware Legal Judgment Prediction  
LA-MGFM: A legal judgment prediction method via sememe-enhanced graph neural networks and multi-graph fusion mechanism  
How Legal Knowledge Graph Can Help Predict Charges for Legal Text  
Zero-shot Transfer of Article-aware Legal Outcome Classification for European Court of Human Rights Cases  
Legal Syllogism Prompting: Teaching Large Language Models for Legal Judgment Prediction  
Methods of incorporating common element characteristics for law article prediction  
An Approach Based on Cross-Attention Mechanism and Label-Enhancement Algorithm for Legal Judgment Prediction  
FCA-LJP: A Method Based on Formal Concept Analysis for Case Judgment Prediction  
融合法律文本结构信息的刑事案件判决预测  
基于注意力机制与知识融合的法律判决预测模型研究

**2022年**  

- [ ] (ACL) [Re11：读论文 EPM Legal Judgment Prediction via Event Extraction with Constraints](https://blog.csdn.net/PolarisRisingWar/article/details/126029464)：这作者代码和数据都没给全，我都不知道别的能复现这篇工作的是咋实现的
- [ ] (EMNLP) Do Charge Prediction Models Learn Legal Theory?
- [x] (IJCAI) [Re28：读论文 CECP Charge Prediction by Constitutive Elements Matching of Crimes](https://blog.csdn.net/PolarisRisingWar/article/details/126484229)
- [ ] (IPM) [Re36：读论文 CEEN Improving legal judgment prediction through reinforced criminal element extraction](https://blog.csdn.net/PolarisRisingWar/article/details/127557195)
- [x] (COLING) [Re 39：读论文 CTM Augmenting Legal Judgment Prediction with Contrastive Case Relations](https://blog.csdn.net/PolarisRisingWar/article/details/127515132)
- [ ] (Artificial Intelligence and Law) [Re41：NumLJP Judicial knowledge‑enhanced magnitude‑aware reasoning for numerical legal judgment predi](https://link.springer.com/article/10.1007/s10506-022-09337-4)

MVE-FLK: A multi-task legal judgment prediction via multi-view encoder fusing legal keywords  
Interpretable prison term prediction with reinforce learning and attention  
Charge prediction modeling with interpretation enhancement driven by double-layer criminal system
Similar Case Based Prison Term Prediction  
Legal Judgment Prediction via Heterogeneous Graphs and Knowledge of Law Articles  
A Computational Intelligence Model for Legal Prediction and Decision Support  
基于BERT模型的多任务法律案件智能判决方法  
基于概念的司法判决预测可解释研究  
基于数据和知识融合的可解释司法判决预测模型  
基于因果推断和多专家FTOPJUDGE机制的法律判决预测方法研究  
基于法条外部知识的法条推荐

**2021年**  
- [ ] (NAACL) [Re18：读论文 GCI Everything Has a Cause: Leveraging Causal Inference in Legal Text Analysis](https://blog.csdn.net/PolarisRisingWar/article/details/126038513)
- [x] (SIGIR) [Re38：读论文 NeurJudge: A Circumstance-aware Neural Framework for Legal Judgment Prediction](https://blog.csdn.net/PolarisRisingWar/article/details/128243315)

Label Definitions Augmented Interaction Model for Legal Charge Prediction  
Equality before the Law: Legal Judgment Consistency Analysis for Fairness  
Mulan: A Multiple Residual Article-Wise Attention Network for Legal Judgment Prediction  
基于法律裁判文书的法律判决预测  
一种法律判决预测的影响因素分析方法  
Dependency Learning for Legal Judgment Prediction with a Unified Text-to-Text Transformer

**2020年**  
1. [x] (ACL) [Re27：读论文 LADAN Distinguish Confusing Law Articles for Legal Judgment Prediction](https://blog.csdn.net/PolarisRisingWar/article/details/126472752)  
LADAN我当年复现的时候出过一点问题，见LADAN文件夹。LADAN官方回复说可以直接改用TF 2实现的D-LADAN（https://github.com/prometheusXN/D-LADAN ），这个我以后可能也会实现一下。

Iteratively Questioning and Answering for Interpretable Legal Judgment Prediction  
An Element-aware Multi-representation Model for Law Article Prediction  
Multi-task Legal Judgement Prediction Combining a Subtask of the Seriousness of Charges  
The Sentencing-Element-Aware Model for Explainable Term-of-Penalty Prediction  
A Relation Learning Hierarchical Framework for Multi-label Charge Prediction  
Legal Judgment Prediction with Label Dependencies

**2019年**
1. [x] (IJCAI) MPBFN [Legal Judgment Prediction via Multi-Perspective Bi-Feedback Network](https://arxiv.org/abs/1905.03969)
3. [x] (ChineseCSCW) MAMD [Charge Prediction for Multi-defendant Cases with Multi-scale Attention](https://link.springer.com/chapter/10.1007/978-981-15-1377-0_59)

Hierarchical Matching Network for Crime Classification
A Recurrent Attention Network for Judgment Prediction
Learning to Predict Charges for Judgment with Legal Graph
Legal Cause Prediction with Inner Descriptions and Outer Hierarchies
Charge Prediction with Legal Attention
Automatic Legal Judgment Prediction via Large Amounts of Criminal Cases
融入罪名关键词的法律判决预测多任务学习模型Multi-task learning model for legal judgment predictions with charge keywords
MANN: A Multichannel Attentive Neural Network for Legal Judgment Prediction
基于胶囊网络的法律罪名预测方法研究

**2018年**
1. [x] (EMNLP) TOPJUDGE [Legal Judgment Prediction via Topological Learning](https://aclanthology.org/D18-1390/)

**2017年**  
1. [ ] (EMNLP) [Re7：读论文 FLA/MLAC/FactLaw Learning to Predict Charges for Criminal Cases with Legal Basis](https://blog.csdn.net/PolarisRisingWar/article/details/125957914)

# 通用文本分类
1. [x] TextCNN
2. [x] Bi-GRU
2. [x] SVM

**2018年**
1. [x] (NAACL) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)
# 结果展示
待补
## 原始格式数据集

## LADAN格式数据集
# 引用
论文还在路上，现在如果大家想引用本GitHub项目可以参考如下格式：
```
@Misc{LJP_Collection,
  title = {LJP_Collection},
  author = {Huijuan Wang},
  howpublished = {\url{https://github.com/PolarisRisingWar/LJP_Collection}},
  year = {2023}
}
```
文字版可参考：Huijuan Wang, LJP_Collection, (2023), GitHub repository, \url{https://github.com/PolarisRisingWar/LJP_Collection}
# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PolarisRisingWar/LJP_Collection&type=Date)](https://star-history.com/#PolarisRisingWar/LJP_Collection&Date)
