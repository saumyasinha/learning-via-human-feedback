###
# Using SGD classfier to do online training (using partial fit)
###

import numpy as np
from sklearn.linear_model import SGDClassifier

## choosing the size of mini-batch
batch_size = 5
## num of features
n_features = 16
## list of class labels
n_classes = [0,1]

## Initialize the SGD classifier with default parameters
clf = SGDClassifier()


def SGD_train(batch_X,batch_y, new_X):
    '''

    :param batch_X: mini-batch of input, shape = (batch_size x n_features)
    :param batch_y: mini-batch of expected output, shape = (1 x batch_size)
    :param new_X: the latest input
    :return: predicted output of new_X
    '''

    clf.partial_fit(batch_X, batch_y, classes=n_classes)
    prediction = clf.predict(new_X)
    return prediction


## "main" would not be needed since the functions from the module will be called directly from the final training script
if __name__ == "__main__":

    for i in range(100):
        batch_X = np.random.rand(batch_size,n_features)
        batch_y = np.random.randint(2, size=batch_size)
        new_X = np.random.rand(1,n_features)
        reward_input_for_tamer = SGD_train(batch_X,batch_y, new_X)





