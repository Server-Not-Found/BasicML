from sklearn import datasets
from matplotlib import pyplot as plt
from kmeans import KMeans

if __name__ == '__main__':
    iris_data = datasets.load_iris()
    data = iris_data['data']
    # 画出有label分类的图
    Y = iris_data['target']
    labels = iris_data['target_names']
    plt.figure(figsize=(15,6))
    plt.subplot(131)
    for iris_type in range(3):
        plt.scatter(data[:,2][Y==iris_type],
                    data[:,3][Y==iris_type],label=labels[iris_type])
    plt.legend()
    plt.title('labeled')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')

    plt.subplot(132)
    plt.scatter(data[:,2],data[:,3])
    plt.title('unlabeled')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')

    num_examples = data.shape[0]
    x_train = data[:,2:4]

    # 指定参数
    num_clusters = 3
    max_iter = 100

    # 通过KMeans模型进行训练
    model = KMeans(x_train,num_clusters)
    centroids, closest_centroids_ids = model.train(max_iter)

    plt.subplot(133)
    plt.title('KMeans')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    for centroid_index, centroid in enumerate(centroids):
        current_examples_index = (closest_centroids_ids == centroid_index).flatten()    # True与False的列表
        plt.scatter(data[:,2][current_examples_index],data[:,3][current_examples_index],label=centroid)
        plt.scatter(centroid[0],centroid[1],color='red',marker='x')
    save_path = './figure/kmeans_demo.jpg'
    plt.savefig(save_path)
    # plt.show()
