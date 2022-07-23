import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from poly import plot_samples, plot_prediction

warnings.filterwarnings('ignore')
gamma_list = [0.1, 1, 100]
C_list = [1, 10, float('inf')]
kernel_list = ['rbf','poly']
X,Y = make_moons(n_samples=100, noise=0.15, random_state=42)

for kernel in kernel_list:
    sub = 330
    plt.figure(figsize=(15, 15))
    plt.suptitle('$Kernel = {}$'.format(kernel), fontsize=20)
    for gamma in gamma_list:
        for C in C_list:
            rbf_svm_clf = Pipeline((
                ('scaler',StandardScaler()),
                ('svm_clf',SVC(kernel=kernel,degree=3,coef0=1,gamma=gamma,C=C))
            ))
            sub+=1
            rbf_svm_clf.fit(X,Y)
            plt.subplot(sub)
            plt.subplots_adjust(hspace=0.5)
            plot_samples(X, Y)
            plot_prediction(rbf_svm_clf, [-2, 3, -2, 3])
            plt.title("$Gamma={}$  $C={}$".format(rbf_svm_clf['svm_clf'].gamma,rbf_svm_clf['svm_clf'].C))
    # plt.show()
    save_path = r'./figures/'+str(kernel)+"_kernel.png"
    plt.savefig(save_path)
