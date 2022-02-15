from sklearn.svm import SVC
import numpy as np
from draw_svm import draw_svm 




train_set = np.load("hw3_material/svm/task2/train_set.npy")
train_labels = np.load("hw3_material/svm/task2/train_labels.npy")

kernels =["linear", "poly", "rbf", "sigmoid"]

for kernel in kernels:
	clf = SVC(kernel=kernel)
	clf.fit(train_set, train_labels)
	draw_svm(clf, train_set, train_labels, train_set[:, 0].min(), train_set[:, 0].max(), train_set[:, 1].min(), train_set[:, 1].max())