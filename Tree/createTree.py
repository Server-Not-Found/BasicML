import matplotlib.pyplot as plt
from math import log
import operator

def createDataset():
    # 根据四个特征判断是否进行贷款
    dataset = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'no'],
               [2, 1, 0, 1, 'no'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']
               ]
    labels = ['F1-AGE','F2-WORK','F3-HOME','F4-LOAN']
    return dataset, labels

# 创建一棵决策树
def createTree(dataset,labels,featLabels,criterion='id3'):
    # 递归结束条件
    classList = [example[-1] for example in dataset]    # 获取分类列表
    if classList.count(classList[0]) == len(classList): # 满纯度
        return classList[0]                             # 返回对应满纯度的分类结果
    if len(dataset[0]) == 1:                            # 已经划分完了
        return majorityCnt(classList)                   # 返回列表中主要的类别

    # 选择最优特征
    bestFeature = chooseBestFeatureToSplit(dataset,criterion) # 0,1,2,3
    bestFeatLabel = labels[bestFeature]
    print("\n"+str(bestFeatLabel))
    featLabels.append(bestFeatLabel)    # 记录特征选择顺序
    myTree = {bestFeatLabel:{}}
    del labels[bestFeature]             # 删除已经选取的特征
    featValList = [example[bestFeature] for example in dataset] # 获取特征值
    unique = set(featValList)
    print(unique)
    print(labels)
    for featVal in unique:
        subLabels = labels[:]   # 复制一份
        # 递归调用，使用嵌套字典来保存决策树
        myTree[bestFeatLabel][featVal] = createTree(splitDataset(dataset,bestFeature,featVal),subLabels,featLabels,criterion)
    return myTree

# 获取结点中数量最多的类别
def majorityCnt(classList):
    classCount={}
    for cls in classList:
        if cls not in classCount.keys():
            classCount[cls]=0
        classCount[cls]+=1
    classCountSorted = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  # 排序操作
    return classCountSorted[0][0]

# 选择最好的一个特征
def chooseBestFeatureToSplit(dataset,criterion):
    numFeature = len(dataset[0])-1
    baseEnt = calcShannoEnt(dataset,criterion)    # 计算根结点熵值
    bestInfoGain = 0                    # 用于记录最大的信息增益
    bestFeature = -1                    # 用于记录最好的划分特征
    # 遍历所有的特征(外层循环遍历的是决策树的每一层)
    for axis in range(numFeature):
        featList = [example[axis] for example in dataset]  # 获取一列的特征值
        uniqueVals = set(featList)  # 获取特征集合
        newEntropy = 0
        # 内层循环遍历的是决策树每一层的每一个分支的结点
        for val in uniqueVals:
            subDataset = splitDataset(dataset,axis,val)    # 不断进行划分并去除已选特征
            prob = len(subDataset)/float(len(dataset))
            newEntropy += prob * calcShannoEnt(subDataset,criterion)  # 权重✖熵值
        infoGain = baseEnt - newEntropy # 计算信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = axis
    return bestFeature

def splitDataset(dataset,axis,val):
    subDataset = []
    # 遍历数据集中所有的样本
    for featVec in dataset:
        if featVec[axis] == val:
            reducedFeatVec = featVec[:axis]         # 去除该样本中对应特征的一列
            reducedFeatVec.extend(featVec[axis+1:])
            subDataset.append(reducedFeatVec)       # subDataset只保存了含有特定特征值的向量
    return subDataset

# 用于计算一个结点的熵值
def calcShannoEnt(dataset,criterion):
    numExample = len(dataset)
    labelCount = {}             # 类别字典
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    if criterion == "c4":  # 信息增益率做法
        IV = 0
        for attr in range(len(dataset[0])-1):
            valCount = {}
            for featVec in dataset:
                if featVec[attr] not in valCount.keys():
                    valCount[featVec[attr]] = 0
                valCount[featVec[attr]] += 1
                for key in valCount:
                    valProp = float(valCount[key]/numExample)
                    IV -= valProp*log(valProp,2)    # 计算固定值

    shannoEnt = 0
    # 遍历所有的类别
    for key in labelCount:
        prop = float(labelCount[key]/numExample)    # 获得了每一个类别对应的概率值
        shannoEnt -= prop*log(prop,2)               # 计算熵值
    if criterion == "c4":
        shannoEnt = shannoEnt/IV
    return shannoEnt


if __name__ == '__main__':
    dataset, labels = createDataset()
    featLabels = []
    myTree = createTree(dataset,labels,featLabels)
    print(myTree)





