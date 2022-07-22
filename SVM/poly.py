import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')

# 画出样本点
def plot_samples(X,Y):
    plt.plot(X[:,0][Y==0],X[:,1][Y==0],'o',color='dodgerblue',marker='s')
    plt.plot(X[:,0][Y==1],X[:,1][Y==1],'o',color='orange',marker='o')
    plt.xlabel('X0')
    plt.ylabel('X1')

def plot_prediction(clf,axes):
    """
    :param clf:     支持向量机分类器实例
    :param axes:    取值范围列表[xmin,xmax,ymin,ymax]
    :return:
    """
    X0s = np.linspace(axes[0],axes[1],100)  # 横坐标一维数组
    X1s = np.linspace(axes[2],axes[3],100)  # 纵坐标一维数组
    x0, x1 = np.meshgrid(X0s,X1s)           # 生成网格坐标矩阵(二维数组)
    # np.c_:列拼接
    # np.ravel()：将多维数组压缩成一维数组
    X = np.c_[x0.ravel(),x1.ravel()]        # 10000行2列，就是10000个坐标
    y_predict = clf.predict(X).reshape(x0.shape)    # 计算出决策边界的坐标
    # 使用等高线图来表示分类板块，因为y_predict的输出无非是0和1，所以两种颜色代表两个分类
    plt.contourf(x0,x1,y_predict,cmap=plt.cm.cividis,alpha=0.3)
    plt.title("$C={}$".format(clf['svm_clf'].C))
    plt.grid()

if __name__ == '__main__':
    # 获取样本点和标签
    X, Y = make_moons(n_samples=100, noise=0.15, random_state=42)

    # 支持向量机流水线构造与运行
    svm_clf1 = LinearSVC(C=10, loss='hinge')
    svm_clf2 = LinearSVC(C=1, loss='hinge')

    polynormial_svm_clf1 = Pipeline((
        ("poly_feature", PolynomialFeatures(degree=3)),  # 升维处理
        ("scaler", StandardScaler()),  # 数据标准化
        ("svm_clf", svm_clf1)  # 支持向量机
    ))

    polynormial_svm_clf2 = Pipeline((
        ("poly_feature", PolynomialFeatures(degree=3)),  # 升维处理
        ("scaler", StandardScaler()),  # 数据标准化
        ("svm_clf", svm_clf2)  # 支持向量机
    ))

    polynormial_svm_clf1.fit(X, Y)
    polynormial_svm_clf2.fit(X, Y)

    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plot_samples(X,Y)
    plot_prediction(polynormial_svm_clf1, [-2, 3, -2, 3])
    plt.subplot(122)
    plot_samples(X, Y)
    plot_prediction(polynormial_svm_clf2,[-2,3,-2,3])
    save_path = './figures/poly.png'
    # plt.show()
    plt.savefig(save_path)