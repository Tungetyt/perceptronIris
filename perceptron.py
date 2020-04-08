import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions
import random
import itertools


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
        # print(f"test{test}")
        return test


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [1, 3]]
    y = iris.target

    # y0 = y.copy()
    # y1 = y.copy()
    # y2 = y.copy()
    mk = Multiklass()
    mk.fit(X, y)
    for xi, target in zip(X, y):
        prediction = mk.predict(xi)
        # print(prediction)
        if prediction == target:
            print(f'{prediction} {target} ok')
        else:
            print(f'{prediction} {target} not ok')
    # ppn = learn_class_with_perceptron(X, y0, 0)
    # ppn1 = learn_class_with_perceptron(X, y1, 1)
    # ppn2 = learn_class_with_perceptron(X, y2, 2)
    #
    # print('give me x')
    # x_to_test = float(input())
    # print('give me x1')
    # x1_to_test = float(input())
    # data_to_test = [[x_to_test, x1_to_test]]
    #
    # corresponding_classes = 0
    # flower_type = -1
    # if ppn.predict(data_to_test) == 1:
    #     corresponding_classes += 1
    #     flower_type = 0
    # if ppn1.predict(data_to_test) == 1:
    #     corresponding_classes += 1
    #     flower_type = 1
    # if ppn2.predict(data_to_test) == 1:
    #     corresponding_classes += 1
    #     flower_type = 2
    #
    # if corresponding_classes != 1:
    #     print('not found')
    # else:
    #     print(f'found: {flower_type}')
    # plot_decision_regions(X=X, y=y, classifier=mk)
    # plt.xlabel(r'$x_1$')
    # plt.ylabel(r'$x_2$')
    # plt.legend(loc='upper left')
    # plt.show()




# def learn_class_with_perceptron(X, y, the_class_to_learn):
#     y[(y != the_class_to_learn)] = -1
#     y[y == the_class_to_learn] = 1
#     ppn = Perceptron(eta=0.1, n_iter=10)
#     ppn.fit(X, y)
#     return ppn
#

class Multiklass:
    def __init__(self):
        n_iter = 20
        eta = 0.1
        self.ppns = [Perceptron(eta=eta, n_iter=n_iter), Perceptron(eta=eta, n_iter=n_iter), Perceptron(eta=eta, n_iter=n_iter)]

    def fit(self, X, y):
        i = 0
        for pps in self.ppns:
            y_norm = y.copy()
            # x_norm = X.copy()
            # c = list(zip(y_norm, x_norm))
            # random.shuffle(c)
            # y_norm, x_norm = zip(*c)
            # print(y_norm)

            # y_norm
            # y_test = []
            # y_test.append(y_norm)
            # print(y_test)

            # yy = []
            # xx =[]
            # for item in y_norm:
            #     yy.extend(item)
            #
            # yy = np.asarray(y_norm)
            #
            # yy = list(itertools.chain.from_iterable(y_norm))

            # for var in y_norm:
            #     yy.append(var)
            #
            # for var in x_norm:
            #     xx.append(var)
            #
            # yyy = np.asarray(yy)
            # xxx = np.asarray(xx)


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
            print(f"data_to_test{data_to_test} corresponding_classes{corresponding_classes} i{i} flower_type{flower_type}")

        if corresponding_classes != 1:
            return -1
        else:
            return flower_type


if __name__ == '__main__':
    main()
