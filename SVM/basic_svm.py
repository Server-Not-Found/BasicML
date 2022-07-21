import numpy as np
import  os
import matplotlib
import matplotlib.pyplot as plt
import warnings
from sklearn import datasets
from sklearn.svm import SVC

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
warnings.filterwarnings('ignore')
"""
鸢尾花数据集
类别：
'setosa'      0
'versicolor'  1
'virginica'   2

150个样本，4个特征
"""

# svm fit之后作图
def plot_svc(svm_clf,xmin,xmax,X,sv=True):
    """
    :param svm_clf:fit后的支持向量机
    :param xmin: 作图的左边界
    :param xmax: 作图的右边界
    :param sv:   判定是否作图
    """
    w = svm_clf.coef_[0]       # 求解支持向量机中的权重向量
    b = svm_clf.intercept_[0]  # 求解支持向量机中的偏置
    # 求解决策超平面
    # w0x0 + w1x1 + b = 1
    x0 = np.linspace(xmin,xmax,200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]    # x1，表示第二个特征
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    if sv:
        # 获取support_vectors_
        if svm_clf.__class__.__name__ == 'LinearSVC':
            decision_function = svm_clf.decision_function(X)
            support_vector_indices = np.where(np.abs(decision_function) == 1 + 1e-15)[0]
            svs = X[support_vector_indices]
        else:
            svs = svm_clf.support_vectors_
        plt.scatter(svs[:,0],svs[:,1],s=130,facecolors='#FF60AF')   # 支持向量

    plt.plot(x0,decision_boundary,'r',linewidth=2)     # 决策边界
    plt.plot(x0,gutter_up,'k--',linewidth=2)            # 正边界
    plt.plot(x0,gutter_down,'k--',linewidth=2)          # 负边界

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris['data'][:,(2,3)]
    # 选择全部样本的第3和第4个特征
    Y = iris['target']          # 对应样本的标签
    setosa_or_versicolor = (Y==0)|(Y==1)
    X = X[setosa_or_versicolor] # 获取第1和第2种类的样本
    Y = Y[setosa_or_versicolor]

    plt.figure(figsize=(14,6))

    # 一般模型
    x0 = np.linspace(0,5.5,200)
    pred_1 = 0.1*x0+0.5
    pred_2 = x0-1.8
    plt.subplot(121)
    plt.plot(x0,pred_1,'r-',linewidth=2)
    plt.plot(x0,pred_2,'m-',linewidth=2)
    plt.plot(X[:,0][Y==1],X[:,1][Y==1],'o',color='dodgerblue',marker='s')
    plt.plot(X[:,0][Y==0],X[:,1][Y==0],'o',color='orange',marker='o')
    plt.axis([0,5.5,0,2])
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.title('Normal Classification')

    # 支持向量机做法
    svm_clf = SVC(kernel='linear',C=float('inf'))
    svm_clf.fit(X,Y)    # 支持向量机计算，得到w和b
    plt.subplot(122)
    plot_svc(svm_clf,0,5.5,X,sv=True)
    plt.plot(X[:,0][Y==1],X[:,1][Y==1],'o',color='dodgerblue',marker='s')
    plt.plot(X[:,0][Y==0],X[:,1][Y==0],'o',color='orange',marker='o')
    plt.axis([0,5.5,0,2])
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.title('Using SVM')

    save_path = r'./figures/basic_svm.png'
    # plt.show()
    plt.savefig(save_path)
