from sklearn.svm import SVC
import numpy as np
from draw_svm import draw_svm 

train_set = np.load("hw3_material/svm/task1/train_set.npy")
train_labels = np.load("hw3_material/svm/task1/train_labels.npy")


c_vals = [0.01, 0.1, 1, 10, 100]
for c in c_vals:
	clf = SVC(kernel='linear', C = c)
	clf.fit(train_set, train_labels)
	draw_svm(clf, train_set, train_labels, train_set[:, 0].min(), train_set[:, 0].max(), train_set[:, 1].min(), train_set[:, 1].max())
