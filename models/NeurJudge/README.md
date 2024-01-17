本项目参考的是NeurJudge官方代码[yuelinan/NeurJudge: The code of "NeurJudge: A Circumstance-aware Neural Framework for Legal Judgment Prediction"(SIGIR2021))](https://github.com/yuelinan/NeurJudge)官方给出的NeurJudge+模型

词向量来自NeurJudge作者

然后分别用small和big数据文件夹传入train_and_test.py（没有使用验证集，直接训练16个epoch然后测试）  
部分辅助数据参考自GitHub项目或直接跟作者邮件要来的，部分是用make_file.py改来的