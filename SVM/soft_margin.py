import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from basic_svm import plot_svc   # 调用basic_svm中的画图方法

iris = datasets.load_iris()
X = iris['data'][:,(2,3)]
# 将0和1作为一类，2作为一类 astype()将布尔值转换成0和1
Y = (iris['target']==2).astype(np.float64)

svm_clf1 = LinearSVC(C=1,random_state=30)
svm_clf2 = LinearSVC(C=100,random_state=30)

scaler1 = StandardScaler()
scaler2 = StandardScaler()

# 构造流水线
svm_clf1_pipe = Pipeline((
    ('std',scaler1),   # 标准化
    ('linear_svc',svm_clf1)
))

svm_clf2_pipe = Pipeline((
    ('std', scaler2),
    ('linear_svc', svm_clf2)
))

svm_clf1_pipe.fit(X,Y)
svm_clf2_pipe.fit(X,Y)
b1 = svm_clf1.decision_function([-scaler1.mean_ / scaler1.scale_])       # 偏置向量1
b2 = svm_clf2.decision_function([-scaler2.mean_ / scaler2.scale_])       # 偏置向量2
w1 = svm_clf1.coef_[0] / scaler1.scale_        # 权重向量2
w2 = svm_clf2.coef_[0] / scaler2.scale_        # 权重向量2
svm_clf1.coef_ = np.array([w1])
svm_clf2.coef_ = np.array([w2])
svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])

plt.figure(figsize=(14,5))
# C=1
plt.subplot(121)
plt.plot(X[:,0][Y==0],X[:,1][Y==0],'o',color='dodgerblue',marker='s',label='Iris-Versicolor')
plt.plot(X[:,0][Y==1],X[:,1][Y==1],'o',color='orange',marker='o',label='Iris-Virginica')
plot_svc(svm_clf1,3,7,X,sv=True)
plt.axis([3,7,0.8,2.6])
plt.title("$C = {}$".format(svm_clf1.C),fontsize=16)
plt.xlabel('X0')
plt.ylabel('X1')
plt.legend(loc='lower right',fontsize=10)

# C=100
plt.subplot(122)
plt.plot(X[:,0][Y==0],X[:,1][Y==0],'o',color='dodgerblue',marker='s',label='Iris-Versicolor')
plt.plot(X[:,0][Y==1],X[:,1][Y==1],'o',color='orange',marker='o',label='Iris-Virginica')
plot_svc(svm_clf2,3,7,X,sv=True)
plt.axis([3,7,0.8,2.6])
plt.title("$C = {}$".format(svm_clf2.C),fontsize=16)
plt.xlabel('X0')
plt.ylabel('X1')
plt.legend(loc='lower right',fontsize=10)

save_path = r'./figures/soft_margin.png'
# plt.show()
plt.savefig(save_path)

