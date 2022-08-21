from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from Bagging import plot_decision_boundary
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

if __name__ == '__main__':
    X,Y = make_moons(n_samples=500,noise=0.3,random_state=42)
    m = len(X)
    plt.figure(figsize=(16,6))
    for subplot,lr in [(121,0.5),(122,1)]:
        sample_weights = np.ones(m)
        plt.subplot(subplot)
        plt.plot(X[:, 0][Y == 0], X[:, 1][Y == 0], 'yo', alpha=0.6)
        plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], 'bs', alpha=0.6)
        for i in range(5):
            svm_clf = SVC(kernel='rbf',C=0.05,random_state=42)
            svm_clf.fit(X,Y,sample_weight=sample_weights)
            y_pred = svm_clf.predict(X)
            sample_weights[y_pred!=Y] *= (1+lr)
            plot_decision_boundary(svm_clf,X,Y,alpha=0.2)
            plt.title('lr={}'.format(lr))
    # plt.show()
    save_path = './figure/AdaBoost.jpg'
    plt.savefig(save_path)