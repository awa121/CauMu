import numpy as np
from sklearn.model_selection import train_test_split

class CANCER(object):
    def __init__(self, mutationData,cancer,pathFileData,replications=10):
        self.path_data=pathFileData
        #data: treatment ,y_factual,x
        self.replications = replications
        # which features are continuous
        self.contfeats = []
        # which features are binary
        self.binfeats = [i for i in range(len(mutationData[0])-2) if i not in self.contfeats]

    def __iter__(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data  + str(i + 1) + '.csv', delimiter=',')
            t, y = data[:, 0].astype(int), data[:, 1][:, np.newaxis].astype(float)
            x =  data[:, 2:].astype(int)
            yield (x, t, y)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + str(i + 1) + '.csv', delimiter=',')
            t, y= data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis]
            x = data[:, 2:]
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr])
            valid = (x[iva], t[iva], y[iva])
            test = (x[ite], t[ite], y[ite])
            yield train, valid, test, self.contfeats, self.binfeats