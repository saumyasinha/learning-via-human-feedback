import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

n_features = 16
classifiers = ["Linear SVM", "RBF SVM", "Random Forest"]


def get_dataset():

    ## needs to be changed, creating a temporary dataset now
    X, y = make_classification(n_features=n_features, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    return X, y

## name to classifier mapping
def classfier_mapping():
    clf_mapping = {
        "Linear SVM" : SVC(kernel="linear", C=0.025),
        "RBF SVM" : SVC(gamma=2, C=1),
        "Random Forest" : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    }

    return clf_mapping


def train(X, y, clf_name):
    '''

    :param X: input with 'n_features' features
    :param y: binary output
    :param clf_name: the name of the classifier used to train the model
    :return: model predictions of input
    '''

    clf= classfier_mapping()[clf_name]

    ## standard prepreocessing
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)

    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)

    print("Validation Accuracy of "+clf_name+":"+str(score))

    predictions = clf.predict(X)

    return predictions


## "main" would not be needed since the functions from the module will be called directly from the final training script
if __name__ == "__main__":

    X,y = get_dataset()

    reward_input_for_tamer = train(X,y,"Linear SVM")





