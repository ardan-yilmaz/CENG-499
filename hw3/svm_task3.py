from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import sys



train_set = np.load("hw3_material/svm/task3/train_set.npy")
train_labels = np.load("hw3_material/svm/task3/train_labels.npy")
nsamples, nx, ny = train_set.shape
train_set = train_set.reshape((nsamples,nx*ny))
#print("reshaped: ", train_set.shape)



#grid_search
def grid_search():
	x_train, x_val, y_train, y_val = train_test_split(train_set, train_labels, test_size=0.2, random_state=1)

	kernels =["linear", "poly", "rbf", "sigmoid"]
	c_vals = [0.01, 0.1, 1, 10, 100]
	gammas = [0.001, 0.01, 0.1, 1]

	for kernel in kernels:
		for c in c_vals:
			for g in gammas:
				clf = SVC(kernel=kernel, C=c, gamma=g)
				clf.fit(x_train, y_train)
				y_pred = clf.predict(x_val)
				print("For kernel ", kernel, " C ", c, " gamma ", g, ", accuracy: ", metrics.accuracy_score(y_val, y_pred))





#TEST:
def test():	
	best_kernel = "rbf"
	best_c = 100
	best_gamma = 0.01

	test_set = np.load("hw3_material/svm/task3/test_set.npy")
	test_labels = np.load("hw3_material/svm/task3/test_labels.npy")
	nsamples, nx, ny = test_set.shape
	test_set = train_set.reshape((nsamples,nx*ny))



	clf = SVC(kernel=best_kernel, C=best_c, gamma=best_gamma)
	clf.fit(train_set, train_labels)
	y_pred = clf.predict(test_set)
	print("For kernel: ", best_kernel, " C: ", best_c, " gamma: ", best_gamma, ", accuracy: ", metrics.accuracy_score(test_labels, y_pred))


if __name__ == "__main__":
	mode = sys.argv[1]

	if mode == "grid_search":
		grid_search()
	elif mode == "test":
		test()



