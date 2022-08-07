# 决策树练习

参考视频：https://www.bilibili.com/video/BV1TZ4y187qX?p=91&amp;vd_source=1eda7e47735eeac0aa46f220d05abb10

本例子中主要实现了基于信息增益以及信息增益率进行结点划分的决策树，可以通过调整参数`criterion`来分别实现`id3`和`c4.5`决策树。

代码的详细分析请看注释。



运行手写代码请执行：

```shell
python createTree.py
```

运行测试代码请执行：

```shell
python test.py
```

## 结果分析

手写代码：

```python
F4-LOAN
{0, 1, 2}
['F1-AGE', 'F2-WORK', 'F3-HOME']

F2-WORK
{0, 1}
['F1-AGE', 'F3-HOME']

F2-WORK
{0, 1}
['F1-AGE', 'F3-HOME']

F1-AGE
{0, 1, 2}
['F3-HOME']
{'F4-LOAN': {0: {'F2-WORK': {0: 'no', 1: 'yes'}}, 1: {'F2-WORK': {0: 'no', 1: {'F1-AGE': {0: 'yes', 1: 'yes', 2: 'no'}}}}, 2: 'yes'}}
```



测试代码：

![](https://github.com/Server-Not-Found/BasicML/blob/master/Tree/figures/tree.png)



通过对比不难发现，测试代码中使用sklearn工具包得出的决策树为一棵二叉树，而手写代码中得出的结果是一棵多叉树。原因是两者的分类标准不同，手写代码中以离散的标签作为分支，而sklearn只提供了二分类的方法。但是两者的分类结果是相同的，只是决策树的结构稍有不同。