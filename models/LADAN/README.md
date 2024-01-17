本文件夹中还需要LADAN官方提供的词向量：cail_thulac.npy

本项目中的代码来源于LADAN官方代码[prometheusXN/LADAN: The source code of article "Distinguish Confusing Law Articles for Legal Judgment Prediction", ACL 2020](https://github.com/prometheusXN/LADAN)
仅使用了LADAN+MTL这一组模型，因为各种后续论文中基本都仅使用该模型作为LADAN模型的baseline

请将环境挪到本文件夹中，然后直接运行LADAN+MTL_small.py文件（这样就不用考虑路径的问题了）
但是LADAN+MTL_large.py这个文件我无法运行，batch size改小后依然会OOM，而且还是过了很久报OOM。但问题就在于我看不懂TensorFlow，看不懂代码，不会改bug，所以就这样吧。

项目中law_label2index_large.pkl这个文件在原项目里不存在，我是通过make_file.py自制的。

本项目使用的环境：
```
numpy 1.21.6
tensorflow-gpu 1.14.0
Keras 2.3.1
cudatoolkit 10.0
cudnn 7.4
jieba 0.42.1
scikit-learn 1.0.2
```