import numpy as np
import pandas as pd
import re
from random import randint
import csv
from flanker import mime
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 用于导出单个邮件的重要信息
def emlAnalyse(email_path):
    emlContent = [] #创建包含发件人，收件人，主题，发送时间，内容等要素的列表
    """
    emlContent的顺序为：
    Sender  Receiver Subject Date Body isSpam
    """
    with open(email_path,'r',encoding='gbk') as f:
        context = f.read()
        # 获取邮件头部信息
        msg = mime.from_string(context)
        headers = msg.headers
        if msg.body is not None:
            body = msg.body.replace('\n','')
        else:
            body = ''
        emlContent.append(headers['From'])      # Sender   0
        emlContent.append(headers['To'])        # Receiver 1
        emlContent.append(headers['Subject'])   # Subject  2
        emlContent.append(headers['Date'])      # Date     3
        emlContent.append(body)                 # Body     4
    return emlContent

# 将eml格式的邮件转化成csv数据
def csvTransformer():
    success = 0
    fail = 0
    print("英文邮件")
    f = open(r'english_email/full/index','r')
    csvfile = open('mailEnglish.csv','w',newline='',encoding='utf-8')
    writer = csv.writer(csvfile)
    for line in f:
        str_list = line.split(" ")
        print(str_list[1].split('\n')[0],end='')

        #垃圾邮件label为0
        if str_list[0] == 'spam':
            label = '0'
        #正常邮件label为1
        elif str_list[0] == 'ham':
            label = '1'
        try:
            emlContent = emlAnalyse('english_email/full/' + str(str_list[1].split("\n")[0]))    #写入路径得到一个包含邮件信息的列表
            if emlContent is not None:
                writer.writerow(
                    [str_list[1].split('\n')[0],emlContent[0], emlContent[1], emlContent[2], emlContent[3], emlContent[4], label])
            success+=1
            print()
        except (UnicodeDecodeError,mime.message.errors.DecodingError,ValueError,TypeError):
            fail+=1
            print('  fail...')
            continue
    rate = success/(success+fail)
    print("数据集利用率："+str(rate))

#从CSV表格中获取邮件主体语料以及邮件对应标签的列表
def get_data(csv_path):
    """
    corpus: list of string(body)
    labels: list of int(labels)
    :param csv_path:
    :return: corpus,labels
    """
    maildf = pd.read_csv(csv_path,header=None, names=['Path','Sender','Receiver','Subject','Date','Body','isSpam'])
    filteredmaildf = maildf[maildf['Body'].notnull()]
    corpus = filteredmaildf['Body']
    labels = filteredmaildf['isSpam']
    corpus = list(corpus)   #邮件主体列表
    labels = list(labels)   #标签列表
    columns_list = maildf.columns
    isSpam_column = columns_list[-1]
    isSpam_data = maildf[isSpam_column]
    # for key,value in isSpam_data.value_counts().items():
    #     print(key,":",value)
    return corpus,labels

#对单个邮件进行单词切分
def textParse(body_text):
    # 按匹配的字串将字符串分割后返回列表,/W+表示匹配任何非单词字符
    ListOfToken = re.split(r' ',body_text.lower())
    return ListOfToken

#所有邮件的集合形成一个二维列表
def normalize_corpus(corpus,tokenize=False):
    normalized_corpus = []
    for text in corpus:
        # 去除重复元素
        filtered_text = list(set(textParse(text))) #一份邮件的主题文本被划分成一个一个词组成的列表
        normalized_corpus.append(filtered_text)
    return normalized_corpus

# 划分训练集与测试集
def dataset_split(corpus,labels,train_rate,test_rate,random_seed):
    train_set, test_set, train_labels, test_labels = train_test_split(corpus,labels,train_size=train_rate,test_size=test_rate,random_state=random_seed)
    normal_train_corpus = normalize_corpus(train_set)
    normal_test_corpus = normalize_corpus(test_set)
    return train_set, test_set, train_labels, test_labels

# 创建语料表
def createVocabList(train_corpus,vocabFilePath):
    # with tqdm(iterable=range(len(train_corpus)), desc="语料表生成进度：", leave=True, unit=' words', unit_scale=True) as pro:
    vocabList = []
    print("针对训练集共"+str(len(train_corpus))+"条邮件生成语料表...")
    for doc in train_corpus:
        docList = textParse(doc)
        for word in docList:
            if word.encode('utf-8').isalpha():   #该词是英文
                vocabList.append(word)
    vocabSet = list(set(vocabList))
    vocabSet.sort(key=vocabList.index)
    vocabSet.insert(0,'net_words')
    with open(vocabFilePath,'w',encoding='utf-8') as vocabFile: # 保存语料表
        for word in vocabSet:
            vocabFile.write(word+"\n")
    print("语料表长度："+str(len(vocabSet)))
    print("语料表已生成...")
    return vocabSet

# 根据语料表生成词向量
def setOfWord2Vec(vocabList,inputString):
    """
    将输入的邮件主体部分字符串转换成词向量
    :param vocabList:输入语料表
    :param inputSet:输入的单条邮件的body
    :return:returnVec包含了语料表长度的01串
    """
    returnVec = [0]*len(vocabList)
    # 遍历邮件
    inputString = textParse(inputString)
    for word in inputString:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        elif "http" or "br" or "html" or "diiv" or "img" or "font" \
                or "tbody" or "td" or "src" in word:
            returnVec[0] += 1
    return returnVec

# 朴素贝叶斯概率计算
def trainNaiveBayes(trainMat,trainLabels):
    """
    :param trainMat:np.array()
    example:
    [
    [0,1,1,....,0],
    .
    .
    [1,0,0,....,1]
    ]
    :param trainLabels:
    example:
    [0,1,0....,1]
    :return:
    p_hamVec,p_spamVec:概率列表
    p_ham:正常邮件的概率
    """
    numTrainDocs = len(trainMat)    #样本个数=邮件个数
    vocabSize = len(trainMat[0])    #语料表的长度
    # 先验概率
    p_ham = sum(trainLabels)/float(numTrainDocs)   #正常邮件的概率
    p_spam = 1-p_ham   # 垃圾邮件的概率
    # 似然分子
    p_ham_Num = np.ones((vocabSize)) #平滑处理,添加一个该属性类别的邮件
    p_spam_Num = np.ones((vocabSize))
    #似然分母
    p_ham_Denom = 2 # 该词出现与不出现两种情况
    p_spam_Denom = 2
    """
    拉普拉斯平滑
    若遇到了训练集中并没有出现的词Di，则平滑处理前P(Di|ham)为0
    拉普拉斯平滑添加了该词的一个样本，因为每个词对于最终是否为垃圾邮件都有一定影响
    因此总体添加的样本为Di|ham和Di|spam
    在本例中，语料表中每个词都作为单独一个属性，所以属性的取值只有出现与不出现两种情况
    """
    for i in range(numTrainDocs):
        if trainLabels[i] == 1: #正常邮件
            p_ham_Num += trainMat[i]    # 统计正常邮件中语料表中的每个词出现的次数+1
            p_ham_Denom += sum(trainMat[i])
        else:
            p_spam_Num += trainMat[i]
            p_spam_Num[-1]=10
            p_spam_Denom += sum(trainMat[i])
    #对数变换，防止因为多个很小的概率相乘导致结果被近似为0
    p_hamVec = np.log(p_ham_Num/p_ham_Denom)
    p_spamVec = np.log(p_spam_Num/p_spam_Denom)
    return p_hamVec,p_spamVec,p_ham

#测试结果
def testNaiveBayes(wordVec,p_hamVec,p_spamVec,p_ham):
    pHam = np.log(p_ham)+sum(wordVec*p_hamVec)
    pSpam = np.log(1.0-p_ham)+sum(wordVec*p_spamVec)
    if pHam>pSpam:
        return 1    #正常邮件
    else:
        return 0    #垃圾邮件

#保存训练结果
def checkpoint(checkpointPath,p_hamVec,p_spamVec,p_ham):
    print("保存模型...")
    checkFile = open(checkpointPath,"w",newline='')
    writer = csv.writer(checkFile)
    writer.writerow(p_hamVec)
    writer.writerow(p_spamVec)
    writer.writerow([p_ham])
    print("存入成功!")

#载入模型
def readCheckFile(checkFilePath):
    checkFile = open(checkFilePath, "r", newline='')
    reader = csv.reader(checkFile)
    for i,row in enumerate(reader):
        if i==0:
            p_hamVec = np.array([np.float64(j) for j in row])
        elif i==1:
            p_spamVec = np.array([np.float64(j) for j in row])
        elif i==2:
            p_ham = np.float64(row[0])
    return p_hamVec,p_spamVec,p_ham

#载入语料表
def readVocab(vocabFilePath):
    with open(vocabFilePath,'r',encoding='utf-8') as vocabFile:
        vocabList = []
        for word in vocabFile:
            vocabList.append(word.replace('\n',''))
    return vocabList
# 垃圾邮件分类器函数
def TrainTestSpam(checkpointPath,train_rate,test_rate,random_seed,vocabList,train_set,train_labels,test_set,test_labels):
    """
    corpus:由get_data函数返回，包含了所有邮件的主体部分
    labels_list:包含所有邮件的标签
    :return:
    """
    trainMat = []   #词向量二维列表
    # training
    with tqdm(iterable=range(len(train_set)), desc="训练进度：", leave=True, unit='条', unit_scale=True) as proBar:
        print("#######Begin Training#######")
        for train_string in train_set:
            trainMat.append(setOfWord2Vec(vocabList,train_string))
            proBar.update(1)
        p_hamVec, p_spamVec, p_ham = \
            trainNaiveBayes(np.array(trainMat), np.array(train_labels))
        checkpoint(checkpointPath,p_hamVec, p_spamVec, p_ham)
    # testing
    right = 0
    index = 0
    with tqdm(iterable=range(len(test_set)), desc="测试进度：", leave=True, unit='条', unit_scale=True) as pro:
        print("#######Begin Testing#######")
        print("测试集大小：" + str(len(test_set)))
        for test_string in test_set:
            wordVec = setOfWord2Vec(vocabList,test_string)
            fore = testNaiveBayes(wordVec,p_hamVec,p_spamVec,p_ham)     #预测结果
            if fore == test_labels[index]:
                right+=1
            index+=1
            pro.update(1)
    accuracy = right / len(test_set)
    print("测试集分类正确率："+str(accuracy))

if __name__ == '__main__':
    
    root_path = r'/home/caijunhong/ml/BayesClassi'
    chinese_email_path = r'/home/caijunhong/ml/BayesClassi/chinese_email'
    english_email_path = r'/home/caijunhong/ml/BayesClassi/english_email'
    
    # 将数据集转换成CSV格式，只需要执行一次
    # csvTransformer()    
    
    trainBeforeTest = 1 # 0表示调用已经训练好的模型 1表示重新训练一次模型
    random_seed = 12    # 划分数据集的随机种子
    train_rate = 0.1    # 训练集占总数据集的比例
    test_rate = 0.01    # 训练集占总数据集的比例
    checkpointPath = root_path + r"/checkpoint_30train.csv"  # 保存模型参数的文件地址
    vocabFilePath = root_path + r'/vocabList.txt'

    corpus,labels_list = get_data(root_path+"//"+"mailEnglish.csv")
    train_set, test_set, train_labels, test_labels \
        = dataset_split(corpus,labels_list,train_rate,test_rate,random_seed)   # 创建训练集，测试集，训练标签，测试标签

    if trainBeforeTest == 1:    # 新训练一个模型并且进行测试
        vocabList = createVocabList(train_set,vocabFilePath)  # 创建语料表
        TrainTestSpam(checkpointPath,train_rate,test_rate,random_seed,vocabList,train_set,train_labels,test_set,test_labels)
    elif trainBeforeTest == 0:  # 载入模型进行测试
        try:
            vocabList = readVocab(vocabFilePath)
            with tqdm(iterable=range(len(test_set)), desc="测试进度：", leave=True, unit='条', unit_scale=True) as pro:
                print("#######Begin Testing#######")
                print("测试集大小：" + str(len(test_set)))
                p_hamVec, p_spamVec, p_ham = readCheckFile(checkpointPath)
                index = 0
                right = 0
                for testDoc in test_set:
                    wordVec = setOfWord2Vec(vocabList,testDoc)
                    fore = testNaiveBayes(wordVec,p_hamVec,p_spamVec,p_ham)
                    if fore == test_labels[index]:
                        right += 1
                    index += 1
                    pro.update(1)
                accuracy = right / len(test_set)
                print("测试集分类正确率：" + str(accuracy))
        except IndexError:
            print("请首先得到一个已经训练好的模型再进行测试！")



