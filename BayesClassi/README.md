# 基于贝叶斯的垃圾邮件分类器

## 1.安装相关依赖

进入根目录，执行以下命令安装所需要导入的包

```shell
pip install -r requirements
```

## 2.清洗数据得到CSV格式的数据集

```python
if __name__ == '__main__':
    # 将数据集转换成CSV格式，只需要执行一次
    csvTransformer()
```

main函数中只用执行`csvTransformer()`函数就能对原生数据集进行处理，获取到后续训练的整体数据集，因为数据集中含有一些无法解码的内容，因此会有一些邮件解码失败，最终数据集的利用率在90%左右。

运行过后会产生`mailEnglish.csv`，即为清洗后的数据，数据结构已经整理成DataFrame了。

## 3.训练与测试

根据自身情况调整好根目录的地址，如下所示：

```python
    root_path = r'/home/caijunhong/ml/BayesClassi'
    chinese_email_path = r'/home/caijunhong/ml/BayesClassi/chinese_email'
    english_email_path = r'/home/caijunhong/ml/BayesClassi/english_email'
```

设置训练模式，数据集划分种子，训练集、测试集比例以及模型参数和语料表的保存路径：

```python
    trainBeforeTest = 1 # 0表示调用已经训练好的模型 1表示重新训练一次模型
    random_seed = 12    # 划分数据集的随机种子
    train_rate = 0.3    # 训练集占总数据集的比例
    test_rate = 0.01    # 训练集占总数据集的比例
    checkpointPath = root_path + r"/checkpoint_30train.csv"  # 保存模型参数的文件地址
    vocabFilePath = root_path + r'/vocabList.txt'
```

第一次训练时，需要将`trainBeforeTest`设置为1，以保存模型参数，后续如果想要调用某个已经训练好的模型，将`trainBeforeTest`设置为0即可，对应的`checkpointPath`为需要调用的模型参数文件路径。

需要注意，设置不同的训练集比例将会生成不一样的语料表，因此需要自行设置语料表的存储地址，以免测试产生偏差。



训练与测试过程样例：

```shell
针对训练集共2365条邮件生成语料表...
语料表长度：23777
语料表已生成...
#######Begin Training#######
训练进度：: 100%|███████████████████████████████████████████████████████████████████████████████▉| 2.36k/2.37k [02:37<00:00, 25.6条/s]保存模型...
存入成功!
#######Begin Testing#######
测试集大小：237
测试进度：: 100%|████████████████████████████████████████████████████████████████████████████████████| 237/237 [00:11<00:00, 20.3条/s]
测试集分类正确率：0.9704641350210971
```

