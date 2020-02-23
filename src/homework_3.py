import copy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Node(object):
    def __init__(self):
        """
        Class for a decision tree node.
        """

        self.left = None
        self.right = None
        self.threshold = None
        self.column = None
        self.depth = None
        self.probas = None
        self.is_terminal = False


class DecisionTreeClassifier(object):
    def __init__(self, max_depth=3, min_samples_leaf=1, min_samples_split=2, impurity='gini'):
        """
        Class for a Decision Tree Classifier.
        Parameters
        ----------
        max_depth : int
            Max depth of a decision tree.
        min_samples_leaf : int
            Minimal number of samples (objects) in a leaf (terminal node).
        min_samples_split : int
            Minimal number of samples (objects) in a node to make a split.
        impurity : str
            Impurity function used for the decision tree building.
        """

        # Make hyper parameters visible inside the class.
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.impurity = impurity

        # Object for the decision tree
        self.Tree = None

        # Helping objects
        self.classes = []
        self.importance = []

    def get_params(self, deep=True):
        """
        Returns class parameters.
        Parameters
        ----------
        deep : boolean
            If True, will return the parameters for this estimator and contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameters of the class.
        """

        params = {'max_depth': self.max_depth,
                  'min_samples_leaf': self.min_samples_leaf,
                  'min_samples_split': self.min_samples_split,
                  'impurity': self.impurity}

        return params

    def set_params(self, **params):
        """
        Set class parameters.
        Parameters
        ----------
        params : dict
            Dictionary of the class parameters.
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def get_importance(self, data):
        return self.importance / len(data)

    def get_leaf_count(self, node):
        value = 0

        if node.is_terminal:
            return 1

        value += self.get_leaf_count(node.right)
        value += self.get_leaf_count(node.left)

        return value

    def node_probabilities(self, y):
        """
        Estimate probabilities of classes in data.
        Parameters
        ----------
        y : numpy.array, shape = (n_objects)
            1D array with the object labels.
            For the classification labels are integers in {0, 1, 2, ...}.
        Returns
        -------
        probas : numpy.array, shape = (n_objects, n_classes)
            2D array with predicted probabilities of each class.
            Example:
                y_predicted_proba = [[0.1, 0.9],
                                     [0.8, 0.2],
                                     [0.0, 1.0],
                                     ...]
        """

        # To store probas.
        probas = []

        # For each class in data ...
        for one_class in self.classes:
            # Estimate probability of the class.
            class_proba = 1.0 * (y == one_class).sum() / len(y)
            # class_proba = 0.8 (example).

            # Store the probability.
            probas.append(class_proba)

        return probas

    @staticmethod
    def gini_calculation(probas):
        """
        Calculate gini value.
        Parameters
        ----------
        probas : numpy.array, shape = (n_objects, n_classes)
            2D array with predicted probabilities of each class.
            Example:
                probas = [0.1, 0.9]
        Returns
        -------
        gini : float
            Gini value.
        """

        gini = 1
        for i in probas:
            gini -= i ** 2

        return gini

    def impurity_calculation(self, y):
        """
        Calculate data impurity.
        Parameters
        ----------
        y : numpy.array, shape = (n_objects)
            1D array with the object labels.
            For the classification labels are integers in {0, 1, 2, ...}.
        Returns
        -------
        impurity : float
            Impurity of the data.
        """

        # Estimate probabilities for each class.
        probas = self.node_probabilities(y)
        # probas = [0.90, 0.10] (example)

        # Calculate impurity of the data.
        if self.impurity == 'gini':
            impurity = self.gini_calculation(probas)
            # impurity = 0.6 (example)

        return impurity

    def best_split(self, X, y):
        """
        Make the best split of data in a decision tree node.
        Parameters
        ----------
        X : numpy.array, shape = (n_objects, n_features)
            Matrix of objects that are described by their input features.
        y : numpy.array, shape = (n_objects)
            1D array with the object labels.
            For the classification labels are integers in {0, 1, 2, ...}.
        Returns
        -------
        best_split_column : int
            Index of the best split column
        best_threshold : float
            The best split condition.
        X_left : numpy.array, shape = (n_objects, n_features)
            Matrix of objects in the left child that are described by their input features.
        y_left : numpy.array, shape = (n_objects)
            1D array with the object labels in the left child.
            For the classification labels are integers in {0, 1, 2, ...}.
        X_right : numpy.array, shape = (n_objects, n_features)
            Matrix of objects in the right child that are described by their input features.
        y_right : numpy.array, shape = (n_objects)
            1D array with labels of the objects in the right child.
            For the classification labels are integers in {0, 1, 2, ...}.
        """

        # To store best split parameters.
        best_split_column = None
        best_threshold = None
        best_information_gain = -999

        # Data impurity before the split.
        impurity = self.impurity_calculation(y)
        # impurity = 0.8 (example)

        # For each column in X.
        for split_column in range(X.shape[1]):
            # Select values of the column.
            x_col = X[:, split_column]
            # x_col = [2.6, 1.3, 0.5, ...] (example)

            # For each value in the column.
            for i_x in range(0, len(x_col)):
                # Take the value as a threshold for a split.
                threshold = x_col[i_x]
                # threshold = 1.3 (example)

                # Make the split into right and left childes.
                y_right = y[x_col > threshold]
                y_left = y[x_col <= threshold]
                # y_left = [0, 1, 1, 0, 1] (example)

                if len(y_right) == 0 or len(y_left) == 0:
                    continue

                # Calculate impurity for each child.
                impurity_left = self.impurity_calculation(y_left)
                impurity_right = self.impurity_calculation(y_right)
                # impurity_right = 0.6 (example)

                # Calculate information gain of the split.
                information_gain = impurity
                information_gain -= impurity_left * len(y_left) / len(y)
                information_gain -= impurity_right * len(y_right) / len(y)
                # information_gain = 0.2 (example)

                # Is this information_gain the best?
                if information_gain > best_information_gain:
                    best_split_column = split_column
                    best_threshold = threshold
                    best_information_gain = information_gain

        # If no split available.
        if best_information_gain == -999:
            return None, None, None, None, None, None

        self.importance[best_split_column] += len(y) * best_information_gain

        # Take the best split parameters and make this split
        x_col = X[:, best_split_column]
        X_left = X[x_col <= best_threshold, :]
        y_left = y[x_col <= best_threshold]
        X_right = X[x_col > best_threshold, :]
        y_right = y[x_col > best_threshold]

        return best_split_column, best_threshold, X_left, y_left, X_right, y_right

    def decision_tree(self, node, X, y):
        """
        Functions builds a decision tree.
        Parameters
        ----------
        node : Node() object
            Current node of the decision tree.
        X : numpy.array, shape = (n_objects, n_features)
            Matrix of objects that are described by their input features.
        y : numpy.array, shape = (n_objects)
            1D array with the object labels.
            For the classification labels are integers in {0, 1, 2, ...}.
        """

        # Check termination conditions.
        if node.depth >= self.max_depth:        # max_depth check.
            node.is_terminal = True
            return
        if len(X) < self.min_samples_split:     # min_samples_split check.
            node.is_terminal = True
            return
        if len(np.unique(y)) == 1:
            node.is_terminal = True
            return

        # Make best split.
        split_column, threshold, X_left, y_left, X_right, y_right = self.best_split(X, y)  # Make a split
        # split_column = 2 (example) column index of the split
        # threshold = 2.74 (example) split_column > threshold

        # Check additional termination conditions.
        if split_column is None:
            node.is_terminal = True
            return
        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:  # min_samples_leaf check
            node.is_terminal = True
            return

        # Add split parameters into the current node.
        node.column = split_column
        node.threshold = threshold

        # Create a left child of the current node.
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.probas = self.node_probabilities(y_left)

        # Create a right child of the current node.
        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.probas = self.node_probabilities(y_right)

        # Make splits for the left and right nodes.
        self.decision_tree(node.right, X_right, y_right)
        self.decision_tree(node.left, X_left, y_left)

    def fit(self, X, y):
        """
        Fit the Decision Tree Classifier.
        Parameters
        ----------
        X : numpy.array, shape = (n_objects, n_features)
            Matrix of objects that are described by their input features.
        y : numpy.array, shape = (n_objects)
            1D array with the object labels.
            For the classification labels are integers in {0, 1, 2, ...}.
        """

        # Get unique class labels.
        self.classes = np.unique(y)
        # self.classes = [0, 1] (example)

        # Create a root node of a decision tree.
        self.Tree = Node()                              # Create an empty node.
        self.Tree.depth = 1                             # The node depth.
        self.Tree.probas = self.node_probabilities(y)   # Init class probabilities.
        self.importance = np.full(X.shape[1], 0, dtype=float)

        # Build the decision tree.
        self.decision_tree(self.Tree, X, y)

    def one_prediction(self, node, x):
        """
        Functions builds a decision tree.
        Parameters
        ----------
        node : Node() object
            Current node of the decision tree.
        x : numpy.array, shape = (n_features,)
            Array of feature values of one object.
        """

        # Termination condition
        if node.is_terminal:    # If it is a leaf (terminal node, no childes).
            return node.probas  # Return probas of the terminal node.
            # node.probas = [0.9, 0.1] (example)

        # Run to the current node's childes.
        # Check split condition. If yes, go to the right child.
        if x[node.column] > node.threshold:
            # Right child.
            probas = self.one_prediction(node.right, x)
            # probas = [0.9, 0.1] (example)
        else:
            # Left child.
            probas = self.one_prediction(node.left, x)
            # probas = [0.9, 0.1] (example)

        return probas

    def predict_proba(self, X):
        """
        Predict class probabilities for unknown objects.
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

        # Create an empty list for predicted probabilities.
        y_predicted_proba = []

        # For each object in X make a prediction.
        for one_x in X:
            # Make the prediction for one object.
            one_proba = self.one_prediction(self.Tree, one_x)
            # one_proba = [0.9, 0.1] (example)

            # Store the predictions.
            y_predicted_proba.append(one_proba)

        return np.array(y_predicted_proba)

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

        # Predict probabilities.
        y_predicted_proba = self.predict_proba(X)
        # y_predicted_proba = [[0.90, 0.10],
        #                      [0.23, 0.77],
        #                       ...]  (example)

        # Find class labels with the highest probability.
        y_predicted = y_predicted_proba.argmax(axis=1)
        # y_predicted = [0, 1] (example)

        return y_predicted

    def remove_leaf(self, node):
        if not node.right.is_terminal:
            if node.right.right.is_terminal and node.right.left.is_terminal:
                if node.right.right.probas[0] >= node.right.left.probas[0]:
                    node.right = cp.deepcopy(node.right.right)
                else:
                    node.right = cp.deepcopy(node.right.left)
                return True

        if not node.left.is_terminal:
            if node.left.right.is_terminal and node.left.left.is_terminal:
                if node.left.right.probas[0] >= node.left.left.probas[0]:
                    node.left = cp.deepcopy(node.left.right)
                else:
                    node.left = cp.deepcopy(node.left.left)
                return True

        if not node.left.is_terminal:
            return self.remove_leaf(node.left)
        elif not node.right.is_terminal:
            return self.remove_leaf(node.right)

        return False


def extract_data():
    return pd.read_csv('../templates/telecom_churn.csv')


def encode_feature(data, name, zero=None, one=None):
    data.loc[:, name] = data[name].replace({zero: 0, one: 1})


def encode_feature_target(data, name):
    return pd.get_dummies(data, columns=[name], prefix_sep='=')


def split_into_samples(data, y_column):
    X_columns = data.columns[data.columns != y_column]

    y = data[y_column].values
    X = data[X_columns].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% for test, 80% for train.
        random_state=42)    # shuffle objects before split.

    return X_train, y_train, X_test, y_test


def task_1():
    data = extract_data()
    print(data.head())

    # Categorical features:
    #   1. area code.
    #   2. international plan.
    #   3. voice mail plan.
    #   4. churn.


def task_2():
    data = extract_data()
    encode_feature(data, 'churn', False, True)
    print(data)


def task_3():
    data = extract_data()
    data = encode_feature_target(data, 'area code')
    encode_feature(data, 'international plan', 'no', 'yes')
    encode_feature(data, 'voice mail plan', 'no', 'yes')
    print(data)


def task_4():
    data = extract_data()

    # Removing unnecessary columns from data matrix.
    targets = [
        'area code',
        'international plan',
        'voice mail plan',
        'number vmail messages',
        'customer service calls',
        'churn'
    ]
    to_drop = list(set(data.columns) - set(targets))
    data = data.drop(to_drop, axis=1)
    del targets
    del to_drop

    # Encoding columns.
    encode_feature(data, 'churn', False, True)
    data = encode_feature_target(data, 'area code')
    encode_feature(data, 'international plan', 'no', 'yes')
    encode_feature(data, 'voice mail plan', 'no', 'yes')
    print(data.head())

    # Creating and printing plots.
    y_column = 'churn'
    x_columns = data.columns[data.columns != y_column]

    plt.figure(figsize=(15, 12))

    for i_col in range(len(x_columns)):
        # Create subplot for each column
        plt.subplot(3, 3, i_col + 1)

        # Get column and label values
        x_col = data[x_columns[i_col]].values
        y_col = data[y_column].values

        # Plot histograms
        bins = np.linspace(0, x_col.max(), 21)
        plt.hist(x_col[y_col == 0], bins=bins, color='r', alpha=0.5, label='0')
        plt.hist(x_col[y_col == 1], bins=bins, color='b', alpha=0.5, label='1')

        # Labels and legend
        plt.xlabel(x_columns[i_col])
        plt.ylabel('Counts')
        plt.legend(loc='best')

    plt.show()

    # I think that 'number vmail messages' and 'customer service calls' are the most informative.


def task_5():
    data = extract_data()

    # Removing unnecessary columns from data matrix.
    targets = [
        'area code',
        'international plan',
        'voice mail plan',
        'number vmail messages',
        'customer service calls',
        'churn'
    ]
    to_drop = list(set(data.columns) - set(targets))
    data = data.drop(to_drop, axis=1)
    del targets
    del to_drop

    # Encoding columns.
    encode_feature(data, 'churn', False, True)
    data = encode_feature_target(data, 'area code')
    encode_feature(data, 'international plan', 'no', 'yes')
    encode_feature(data, 'voice mail plan', 'no', 'yes')

    X_train, y_train, X_test, y_test = split_into_samples(data, 'churn')
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)


def task_6():
    data = extract_data()

    # Removing unnecessary columns from data matrix.
    targets = [
        'area code',
        'international plan',
        'voice mail plan',
        'number vmail messages',
        'customer service calls',
        'churn'
    ]
    to_drop = list(set(data.columns) - set(targets))
    data = data.drop(to_drop, axis=1)
    del targets
    del to_drop

    # Encoding columns.
    encode_feature(data, 'churn', False, True)
    data = encode_feature_target(data, 'area code')
    encode_feature(data, 'international plan', 'no', 'yes')
    encode_feature(data, 'voice mail plan', 'no', 'yes')

    X_train, y_train, X_test, y_test = split_into_samples(data, 'churn')

    dtc = DecisionTreeClassifier(max_depth=30, min_samples_leaf=1, min_samples_split=2, impurity='gini')
    dtc.fit(X_train, y_train)
    prediction = dtc.predict(X_test)

    print("Accuracy is", accuracy_score(y_test, prediction))


def task_7():
    data = extract_data()

    # Removing unnecessary columns from data matrix.
    targets = [
        'area code',
        'international plan',
        'voice mail plan',
        'number vmail messages',
        'customer service calls',
        'churn'
    ]
    to_drop = list(set(data.columns) - set(targets))
    data = data.drop(to_drop, axis=1)
    del targets
    del to_drop

    # Encoding columns.
    encode_feature(data, 'churn', False, True)
    data = encode_feature_target(data, 'area code')
    encode_feature(data, 'international plan', 'no', 'yes')
    encode_feature(data, 'voice mail plan', 'no', 'yes')

    X_train, y_train, X_test, y_test = split_into_samples(data, 'churn')

    dtc = DecisionTreeClassifier(max_depth=30, min_samples_leaf=1, min_samples_split=2, impurity='gini')
    dtc.fit(X_train, y_train)

    print("Importance:", *dtc.get_importance(y_train))

    # The most informative feature for classification is 'customer service calls'.


def task_8():
    data = extract_data()

    # Removing unnecessary columns from data matrix.
    targets = [
        'area code',
        'international plan',
        'voice mail plan',
        'number vmail messages',
        'customer service calls',
        'churn'
    ]
    to_drop = list(set(data.columns) - set(targets))
    data = data.drop(to_drop, axis=1)
    del targets
    del to_drop

    # Encoding columns.
    encode_feature(data, 'churn', False, True)
    data = encode_feature_target(data, 'area code')
    encode_feature(data, 'international plan', 'no', 'yes')
    encode_feature(data, 'voice mail plan', 'no', 'yes')

    X_train, y_train, X_test, y_test = split_into_samples(data, 'churn')

    dtc = DecisionTreeClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=2, impurity='gini')
    dtc.fit(X_train, y_train)

    prediction = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    print('Old accuracy is ', accuracy)

    while True:
        dtc_copy = cp.deepcopy(dtc)
        is_removed = dtc_copy.remove_leaf(dtc_copy.Tree)

        if not is_removed:
            break

        y_test_predict = dtc_copy.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_test_predict)

        if accuracy_test >= accuracy:
            accuracy = accuracy_test
            dtc = dtc_copy
        else:
            break

    print('New accuracy is ', accuracy)


if __name__ == '__main__':
    task_8()
