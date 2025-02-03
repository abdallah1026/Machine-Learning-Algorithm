import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from LogisticRegression import LogisticRegression

bc = load_breast_cancer()

X,y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.2, random_state = 42)

clf = LogisticRegression(lr=0.02)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

def accuracy(y_test, y_pred):

    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, y_pred)

print(f"Accuracy: {acc}")