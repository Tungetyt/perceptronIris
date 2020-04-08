import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


class wieloklas:
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [1, 3]]
    y = iris.target

    y0 = y.copy()
    y1 = y.copy()
    y2 = y.copy()
    lrgd = learn_class_with_regression(X, y0, 0)
    lrgd1 = learn_class_with_regression(X, y1, 1)
    lrgd2 = learn_class_with_regression(X, y2, 2)
    print('give me x')
    x_to_test = float(input())
    print('give me x1')
    x1_to_test = float(input())
    data_to_test = [[x_to_test, x1_to_test]]

    probabilities = calc_probabilities(lrgd, lrgd1, lrgd2, data_to_test)
    print(probabilities)

    print(f'found: {probabilities.index(max(probabilities))}')




def learn_class_with_regression(X, y, the_class_to_learn):
    y[(y != the_class_to_learn)] = -3  #tymczasowo -3
    y[y == the_class_to_learn] = 1
    y[y == -3] = 0    #teraz już 0
    lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd.fit(X, y)
    return lrgd


def calc_probabilities(lrgd, lrgd1, lrgd2, data_to_test):
    return [lrgd.activation(lrgd.net_input(data_to_test)),
            lrgd1.activation(lrgd1.net_input(data_to_test)),
            lrgd2.activation(lrgd2.net_input(data_to_test))]


if __name__ == '__main__':
    main()


