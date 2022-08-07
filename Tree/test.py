from  sklearn import tree
from matplotlib import pyplot as plt

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

X = [vec[:4] for vec in dataset]
Y = [vec[-1] for vec in dataset]

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X,Y)
tree.plot_tree(clf)
# plt.show()
save_path = r'./figures/'+"tree.png"
plt.savefig(save_path)
