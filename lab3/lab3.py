import numpy as np
import os
from mlxtend.data import loadlocal_mnist
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt

class Detector:

    def __init__(self):
        np.random.seed(42)
        # To plot pretty figures
        mpl.rc('axes', labelsize=14)
        mpl.rc('xtick', labelsize=12)
        mpl.rc('ytick', labelsize=12)
        # Where to save the figures
        #PROJECT_ROOT_DIR = "."
        #IMAGE_DIR = "FIXME"
       
        

    # def save_fig(self,fig_id, tight_layout=True):
    #     path = os.path.join(PROJECT_ROOT_DIR, "images", IMAGE_DIR, fig_id + ".png")
    #     print("Saving figure", fig_id)
    #     if tight_layout:
    #         plt.tight_layout()
    #     plt.savefig(path, format='png', dpi=300)
        

    # def random_digit(self):
    #     import matplotlib as mpl
    #     import matplotlib.pyplot as plt 
    #     some_digit = X[36000]
    #     some_digit_image = some_digit.reshape(28, 28)
    #     plt.imshow(some_digit_image, cmap = mpl.cm.binary,
    #             interpolation="nearest")
    #     plt.axis("off")

    #     save_fig("some_digit_plot")
    #     plt.show()

    
    def load_and_sort(self):
        try:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1, cache=True)
            #mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
            #self.sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
        except ImportError:
            from sklearn.datasets import fetch_mldata
            mnist = fetch_mldata('MNIST original')
        return mnist["data"], mnist["target"]
        

    def sort_by_target(self,mnist):
        reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
        reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
        mnist.data[:60000] = mnist.data[reorder_train]
        mnist.target[:60000] = mnist.target[reorder_train]
        mnist.data[60000:] = mnist.data[reorder_test + 60000]
        mnist.target[60000:] = mnist.target[reorder_test + 60000]


    def train_predict(self,X,y,some_digit):  
        X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
        shuffle_index = np.random.permutation(60000)
        X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

        # Example: Binary number 4 Classifier
        # y_train_4 = (y_train == 4)
        # y_test_4 = (y_test == 4)

        # Number-5 detector binary classifier
        y_train = y_train.astype(np.int8)
        y_test = y_test.astype(np.int8)
        y_train_5 = (y_train == 5)
        y_test_5 = (y_test == 5)
        
        # print prediction result of the given input some_digit
        #some_digit = X[36000]
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(StandardScaler(),
           SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(X_train, y_train_5)
        print(clf.predict([some_digit]))
        self.calculate_cross_val_score(clf,X_train,y_train_5)
        
        
    def calculate_cross_val_score(self,sgd_clf,X_train,y_train_5):
        from sklearn.model_selection import cross_val_score
        val=cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
        print(val)

if __name__ == "__main__":
    obj=Detector()
    # X, y = loadlocal_mnist(
    #     images_path='train-images-idx3-ubyte', 
    #     labels_path='train-labels-idx1-ubyte')
    # #print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
    # #print('\n1st row', X[0])
    # #print('Digits:  0 1 2 3 4 5 6 7 8 9')
    # #print('labels: %s' % np.unique(y))
    # #print('Class distribution: %s' % np.bincount(y))
    # np.savetxt(fname='images.csv', 
    #        X=X, delimiter=',', fmt='%d')
    # np.savetxt(fname='labels.csv', 
    #        X=y, delimiter=',', fmt='%d')
    X,y=obj.load_and_sort()
    #print(X)
    #print(y)
    X=X.values
    obj.train_predict(X,y,X[36000])