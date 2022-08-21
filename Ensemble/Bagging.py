from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# 单个分类器的分类正确度
# for clf in (log_clf,svm_clf,rnd_clf):
#     clf.fit(X_train,Y_train)
#     Y_predict = clf.predict(X_test)
#     print(clf.__class__.__name__,accuracy_score(Y_test,Y_predict))

# 投票分类器
# voter.fit()要求每个学习器都能得到一个概率值
# 硬投票实验
# print("hard voting")
# voter = VotingClassifier(estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
#                  voting='hard')
# voter.fit(X_train,Y_train)
# Y_predict = voter.predict(X_test)
# print(voter.__class__.__name__,accuracy_score(Y_test,Y_predict))
#
# print("soft voting")
# voter = VotingClassifier(estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
#                  voting='soft')
# voter.fit(X_train,Y_train)
# voter_predict = voter.predict(X_test)
# print(voter.__class__.__name__,accuracy_score(Y_test,voter_predict))
#
# bag_clf.fit(X_train,Y_train)
# bag_predict = bag_clf.predict(X_test)
# print(bag_clf.__class__.__name__,accuracy_score(Y_test,bag_predict))

def plot_decision_boundary(clf,X,Y,alpha=0.6,axes=[-2,3,-1.5,2],contour=True):
    x = np.linspace(axes[0],axes[1],100)
    y = np.linspace(axes[2],axes[3],100)
    x_mesh, y_mesh = np.meshgrid(x,y)
    point_List = np.c_[x_mesh.ravel(),y_mesh.ravel()]
    y_pred = clf.predict(point_List).reshape(x_mesh.shape)
    custom_cmap2 = ListedColormap(['#FFFF00','#FF3333','#33FFFF'])
    plt.contourf(x_mesh, y_mesh, y_pred, cmap=custom_cmap2, alpha=alpha)
    if contour:
        custom_cmap2 = ListedColormap(['#FFFF00','#FF3333','#33FFFF'])
        plt.contour(x_mesh, y_mesh, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.axis(axes)
    plt.xlabel('x1')
    plt.ylabel('x2')

if __name__ == '__main__':
    X,Y = make_moons(n_samples=500,noise=0.3,random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42)
    log_clf = LogisticRegression(random_state=42)
    svm_clf = SVC(random_state=42,probability=True)
    rnd_clf = RandomForestClassifier(random_state=42)
    bag_clf = BaggingClassifier(n_estimators=500,max_samples=100,n_jobs=-1,random_state=42)
    tree_clf = DecisionTreeClassifier(random_state=42)

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(X[:, 0][Y == 0], X[:, 1][Y == 0], 'yo', alpha=0.6)
    plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], 'bs', alpha=0.6)
    tree_clf.fit(X_train,Y_train)
    plot_decision_boundary(tree_clf,X,Y)
    plt.title('Decision Tree')
    plt.subplot(122)
    plt.plot(X[:, 0][Y == 0], X[:, 1][Y == 0], 'yo', alpha=0.6)
    plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], 'bs', alpha=0.6)
    bag_clf.fit(X_train,Y_train)
    plot_decision_boundary(bag_clf,X,Y)
    plt.title('Decision Tree with Bagging')
    # plt.show()
    save_path = './figure/Bagging.jpg'
    plt.savefig(save_path)
