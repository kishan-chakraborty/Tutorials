"""
    Implementing simple linear regression using gradient descent
"""
import numpy as np
from sklearn.datasets import make_regression


class SimpleLinearRegression():
    """
        Applying gradient descent in simple linear regression
    """
    def fit(self, train_x, train_y, l_rate=0.001, n_epochs=10):
        """
            lr = learning rate of gradient descent
            n_epochs = No. of epochs
        """
        self.x_train = train_x
        self.y_train = train_y
        self.l_rate = l_rate
        self.n_epochs = n_epochs

        [m, b] = [200, -29]  # random initialization of slope and intercept
        i = 0
        while i <= self.n_epochs:
            dl_dm = -2 * (np.sum((self.y_train - m * self.x_train - b) * self.x_train))
            dl_db = -2 * (np.sum(self.y_train - m * self.x_train - b))
            [m, b] = [m, b] - self.l_rate * np.array([dl_dm, dl_db])
            print([m, b])
            i += 1


if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=13)
    lr = SimpleLinearRegression()
    lr.fit(X, y, 0.0003, 100)
