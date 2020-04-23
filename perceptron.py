import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        test = np.where(self.net_input(X) >= 0.0, 1, -1)
        return test


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [1, 3]]
    y = iris.target

    mk = Multiklass()
    mk.fit(X, y)
    for xi, target in zip(X, y):
        prediction = mk.predict(xi)
        if prediction == target:
            print(f'{prediction} {target} ok')
        else:
            print(f'{prediction} {target} not ok')


class Multiklass:
    def __init__(self):
        n_iter = 20
        eta = 0.1
        self.ppns = [Perceptron(eta=eta, n_iter=n_iter), Perceptron(eta=eta, n_iter=n_iter),
                     Perceptron(eta=eta, n_iter=n_iter)]

    def fit(self, X, y):
        i = 0
        for pps in self.ppns:
            y_norm = y.copy()
            y_norm[(y_norm != i)] = -1
            y_norm[y_norm == i] = 1
            pps.fit(X, y_norm)
            print(y_norm)
            i = i + 1

    def predict(self, data_to_test):
        i = 0
        flower_type = -1
        corresponding_classes = 0

        for ppn in self.ppns:
            if ppn.predict(data_to_test) == 1:
                corresponding_classes += 1
                flower_type = i
            i = i + 1
            print(
                f"data_to_test{data_to_test} corresponding_classes{corresponding_classes} i{i} flower_type{flower_type}")

        if corresponding_classes != 1:
            return -1
        else:
            return flower_type


if __name__ == '__main__':
    main()
