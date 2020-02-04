import numpy as np
from scipy.spatial import cKDTree
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split


class KNNClassifier(object):
    def __init__(self, max_dist=1.0, use_kd_tree=False, use_weights=False):
        """
        This is a constructor of the class.
        Here you can define parameters (max_dist) of the class and
        attributes, that are visible within all methods of the class
        Parameters
        ----------
        max_dist : float
            Maximum distance between an object and its neighbors.
        """
        self.max_dist = max_dist
        self.use_kd_tree = use_kd_tree
        self.use_weights = use_weights
        self.X_train = None
        self.y_train = None

    @staticmethod
    def get_distances(x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=1))

    @staticmethod
    def get_weights(distances):
        weights = np.zeros(len(distances), dtype=np.float32)
        result = 0.0

        for i in range(len(distances)):
            weights[i] += 1.0 / distances[i]
            result += 1.0 / distances[i]

        return weights / result

    def fit(self, X, y):
        """
        This method trains the KNN classifier.
        Actually, the KNN classifier has no training procedure.
        It just remembers data (X, y) that will be used for predictions.
        Parameters
        ----------
        X : numpy.array, shape = (n_objects, n_features)
            Matrix of objects that are described by their input features.
        y : numpy.array, shape = (n_objects)
            1D array with the object labels.
            For the classification labels are integers in {0, 1, 2, ...}.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        This methods performs labels prediction for new objects.
        Parameters
        ----------
        X : numpy.array, shape = (n_objects, n_features)
            Matrix of objects that are described by their input features.
        Returns
        -------
        y_predicted : numpy.array, shape = (n_objects)
            1D array with predicted labels.
            For the classification labels are integers in {0, 1, 2, ...}.
        """

        # Create an empty list for predicted labels.
        y_predicted = []

        # Prediction under here.
        for i in X:
            if self.use_kd_tree:
                kd_tree = cKDTree(self.X_train, leafsize=30)
                n_indexes = kd_tree.query_ball_point(i, self.max_dist)
                n_distances = KNNClassifier.get_distances(kd_tree.data[n_indexes], i)
            else:
                distances = KNNClassifier.get_distances(self.X_train, i)
                sorted_d = distances.argsort()
                under_max_d = distances[distances - self.max_dist <= 0.000001]

                if len(under_max_d) <= 0:
                    n_indexes = sorted_d[0]
                else:
                    n = len(under_max_d) + 1
                    n_indexes = sorted_d[:n]

                n_distances = distances[n_indexes]

            n_labels = self.y_train[n_indexes]

            if self.use_weights:
                weights = KNNClassifier.get_weights(n_distances)
                items = {}

                for j in range(len(n_labels)):
                    label_i = n_labels[j]

                    if label_i in items:
                        items[label_i] += weights[j] * 1.0
                    else:
                        items[label_i] = weights[j] * 1.0

                maximum = max(items.values())

                for j in items:
                    if items[j] == maximum:
                        y_predicted.append(j)
                        break
            else:
                unique_labels, label_counts = np.unique(n_labels, return_counts=True)
                label_max_count = unique_labels[label_counts == label_counts.max()][0]
                y_predicted.append(label_max_count)

        return np.array(y_predicted)

    def predict_proba(self, X):
        """
        This methods performs prediction of probabilities of each class for new objects.
        Parameters
        ----------
        X : numpy.array, shape = (n_objects, n_features)
            Matrix of objects that are described by their input features.
        Returns
        -------
        y_predicted_proba : numpy.array, shape = (n_objects, n_classes)
            2D array with predicted probabilities of each class.
            Example:
                y_predicted_proba = [[0.1, 0.9],
                                     [0.8, 0.2],
                                     [0.0, 1.0],
                                     ...]
        """

        # Create an empty list for predictions.
        y_predicted_proba = []

        for i in X:
            if self.use_kd_tree:
                kd_tree = cKDTree(self.X_train, leafsize=30)
                n_indexes = kd_tree.query_ball_point(i, self.max_dist)
                n_distances = KNNClassifier.get_distances(kd_tree.data[n_indexes], i)
            else:
                distances = KNNClassifier.get_distances(self.X_train, i)
                sorted_d = distances.argsort()
                under_max_d = distances[distances - self.max_dist <= 0.000001]

                if len(under_max_d) <= 0:
                    n_indexes = sorted_d[0]
                else:
                    n = len(under_max_d) + 1
                    n_indexes = sorted_d[:n]

                n_distances = distances[n_indexes]

            n_labels = self.y_train[n_indexes]

            if self.use_weights:
                weights = self.get_weights(n_distances)
                items = {}

                for j in range(len(n_labels)):
                    label_i = n_labels[j]

                    if label_i in items:
                        items[label_i] += weights[j] * 1.0
                    else:
                        items[label_i] = weights[j] * 1.0

                proba = np.zeros(2)
                item_count = sum(items.values())

                for j in items:
                    proba[j] = items[j] / item_count

                y_predicted_proba.append(proba)
            else:
                unique_labels, label_counts = np.unique(n_labels, return_counts=True)
                proba = np.zeros(2)
                item_count = label_counts.sum()

                for j in unique_labels:
                    proba[j] = label_counts[unique_labels.index(j)] / item_count

                y_predicted_proba.append(proba)

        return np.array(y_predicted_proba)


def task_1():
    X, y = make_moons(n_samples=1000, random_state=42, noise=0.2)
    answer = np.array([
        [-0.112,  0.52],
        [1.143, -0.343]
    ])

    assert np.array_equal(np.round(X[:2], 3), answer), ('Check your solution. Task 1.')
    return X, y


def task_2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)
    answer = np.array([
        [0.77, -0.289],
        [0.239, 1.041]
    ])

    assert np.array_equal(np.round(X_train[:2], 3), answer), ('Check your solution. Task 2.')
    return X_train, X_test, y_train, y_test


def task_3(X_train, X_test, y_train, y_test):
    knn = KNNClassifier(max_dist=0.5)
    knn.fit(X_train, y_train)
    y_test_predict = knn.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_predict)

    print('Test accurancy of KNN classifier: ', accuracy_test)

    assert accuracy_test == 0.964, ('Check your solution. Task 3.')


def task_4(X_train, X_test, y_train, y_test):
    knn = KNNClassifier(max_dist=0.5, use_kd_tree=True)
    knn.fit(X_train, y_train)
    y_test_predict = knn.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_predict)

    print("Test accuracy of KNN classifier: ", accuracy_test)

    assert accuracy_test == 0.964, ('Check your solution. Task 4.')


def task_5(X_train, X_test, y_train, y_test):
    knn = KNNClassifier(max_dist=0.5, use_kd_tree=True, use_weights=True)
    knn.fit(X_train, y_train)
    y_test_predict = knn.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_predict)

    print("Test accuracy of KNN classifier: ", accuracy_test)

    assert accuracy_test == 0.968, ('Check your solution. Task 5.')


def task_6(X_train, X_test, y_train):
    knn = KNNClassifier(max_dist=0.5, use_kd_tree=True, use_weights=True)
    knn.fit(X_train, y_train)
    y_test_predict_proba = knn.predict_proba(X_test)
    answer = np.array([
        [0.046, 0.954],
        [0.962, 0.038]
    ])

    assert np.array_equal(np.round(y_test_predict_proba[:2], 3), answer), ('Check your solution. Task 6.')


if __name__ == '__main__':
    X, y = task_1()
    X_train, X_test, y_train, y_test = task_2(X, y)
    task_3(X_train, X_test, y_train, y_test)
    task_4(X_train, X_test, y_train, y_test)
    task_5(X_train, X_test, y_train, y_test)
    task_6(X_train, X_test, y_train)