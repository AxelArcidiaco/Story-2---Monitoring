# Loading required Python libraries
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import numpy as np
import unittest

class TestSVC(unittest.TestCase):
    def setUp(self):
        # Loading digits dataset from sklearn
        self.data = datasets.load_breast_cancer()

        # Features
        self.X = self.data.data

        # Target
        self.y = self.data.target

        # train-test split of the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=0
        )

        # Loading classifier
        self.sklearn_classifier = SVC()

    def test_sklearn_classifier(self):
        self._test_classifier(self.sklearn_classifier)

    def _test_classifier(self, classifier):
        classifier.fit(self.X_train, self.y_train)
        y_pred = classifier.predict(self.X_test)

        # Assertion tests
        self.assertTrue(confusion_matrix(self.y_test, y_pred) is not None)
        self.assertTrue(classification_report(self.y_test, y_pred) is not None)

    def test_learning_curve(self):
        self._test_learning_curve(self.sklearn_classifier)

    def _test_learning_curve(self, classifier):
        N, train_score, val_score = learning_curve(
            classifier, self.X_train, self.y_train, cv=4, scoring='f1', train_sizes=np.linspace(0.1, 1, num=10)
        )

        # Assertion tests
        self.assertTrue(N is not None)
        self.assertTrue(train_score is not None)
        self.assertTrue(val_score is not None)

if __name__ == '__main__':
    unittest.main()