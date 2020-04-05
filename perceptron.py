import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions


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
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [1, 3]]
    y = iris.target
    # print(iris.data)
    # print(y)
    # print(X)
    y0 = y.copy()
    y1 = y.copy()
    y2 = y.copy()

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    # print(X_train)
    # print(y_train)
    # X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
    # y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    #
    # w perceptronie wyjście jest albo 1 albo -1    
    # y_train_01_subset[(y_train_01_subset == 0)] = -1
    ppn = learn_class_with_perceptron(X, y0, 0)
    ppn1 = learn_class_with_perceptron(X, y1, 1)
    ppn2 = learn_class_with_perceptron(X, y2, 2)

    # plot_decision_regions(X=X, y=y1, classifier=ppn1)
    # plt.xlabel(r'$x_1$')
    # plt.ylabel(r'$x_2$')
    # plt.legend(loc='upper left')
    # plt.show()

    # sum_ = 0
    # ok = 0
    # notok = 0
    # for iks in X:
    #     sum_ += ppn.predict(iks)
    #     sum_ += ppn1.predict(iks)
    #     sum_ += ppn2.predict(iks)
    #     if sum_ == -1:
    #         ok += 1
    #     else:
    #         notok += 1
    #     sum_ = 0

    # print(y)
    # sum_ = 0
    #     ok = 0
    #     notok = 0
    #     isfound = 0
    #     for iks, target in zip(X, y):
    #         if ppn.predict(iks) == target:
    #              isfound += 1
    #         if ppn1.predict(iks) == target:
    #              isfound += 1
    #         if ppn2.predict(iks) == target:
    #              isfound += 1
    #         if isfound == 1:
    #             ok += 1
    #         else:
    #             notok += 1
    #         print(f'{ppn.predict(iks)}{target}')
    #         isfound = 0
    #
    #     print(f'ok: {ok}')
    #     print(f'notok: {notok}')
    # #     testowanie usera!!!!!!!!!!!!!!!!
    #
    print('give me x')
    x_to_test = float(input())
    print('give me x1')
    x1_to_test = float(input())
    data_to_test = [[x_to_test, x1_to_test]]
    # print(X)
    # print(data_to_test)
    # print(ppn.predict(data_to_test))

    corresponding_classes = 0
    flower_type = -1
    if ppn.predict(data_to_test) == 1:
        corresponding_classes += 1
        flower_type = 0
    if ppn1.predict(data_to_test) == 1:
        corresponding_classes += 1
        flower_type = 1
    if ppn2.predict(data_to_test) == 1:
        corresponding_classes += 1
        flower_type = 2

    if corresponding_classes != 1:
        print('not found')
    else:
        print(f'found: {flower_type}')


def learn_class_with_perceptron(X, y, the_class_to_learn):
    y[(y != the_class_to_learn)] = -1
    y[y == the_class_to_learn] = 1
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    return ppn


if __name__ == '__main__':
    main()
