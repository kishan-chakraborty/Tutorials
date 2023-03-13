"""
    Implementing simple linear regression using gradient descent
"""
import numpy as np
from sklearn.datasets import make_regression


class LinearRegression():
    """
        Applying gradient descent in simple linear regression
    """
    def batch_gradient_descent(self, train_x, train_y, l_rate=0.001, n_epochs=10):
        """
            lr = learning rate of gradient descent
            n_epochs = No. of epochs
        """
        self.x_train = train_x
        self.y_train = train_y
        self.l_rate = l_rate
        self.n_epochs = n_epochs

        self.n_sample, self.n_features = self.x_train.shape
        self.beta0_batch = np.random.rand(1)  # random initialization of intercept
        self.beta_batch = np.random.rand(self.n_features)  # random initialization of slope
        print(self.beta_batch)
        i = 0
        while i <= self.n_epochs:
            y_hat = np.dot(self.x_train, self.beta_batch) + self.beta0_batch
            dl_dbeta0 = (-1 / self.n_sample) * np.sum(self.y_train - y_hat)
            dl_dbeta = (-1 / self.n_sample) * np.dot(np.transpose(self.x_train), (y - y_hat))

            self.beta0_batch = self.beta0_batch - self.l_rate * dl_dbeta0
            self.beta_batch = self.beta_batch - self.l_rate * dl_dbeta

            print(self.beta_batch)
            i += 1

    def stochastic_gradient_descent(self):
        """
            Applying stochastic gradient descent to obtain the weights
        """
        self.beta0_stoch = np.random.rand(1)  # random initialization of intercept
        self.beta_stoch = np.random.rand(self.n_features)  # random initialization of slope
        print(self.beta_stoch)
        i = 0
        while i <= self.n_epochs:
            idxs = np.random.choice(np.arange(0, self.n_sample), size=self.n_sample, replace=False)
            for idx in idxs:
                y_hat = np.dot(self.x_train[idx], self.beta_stoch) + self.beta0_stoch
                dl_dbeta0 = -2 * (self.y_train[idx] - y_hat)
                dl_dbeta = -2 * self.x_train[idx] * (self.y_train[idx] - y_hat)

                self.beta0_stoch = self.beta0_stoch - self.l_rate * dl_dbeta0
                self.beta_stoch = self.beta_stoch - self.l_rate * dl_dbeta

            print(self.beta_stoch)
            i += 1

    def mini_batch_gradient_descent(self, batch_size):
        """
            Implementing Mini Batch Gradient descent to solve the linear regression problem
        """
        self.beta0_mini = np.random.rand(1)  # random initialization of intercept
        self.beta_mini = np.random.rand(self.n_features)  # random initialization of slope
        print(self.beta_mini)
        i = 0
        while i <= self.n_epochs:
            batches = np.random.choice(np.arange(0, self.n_sample), size=(int(self.n_sample / batch_size), batch_size), replace=False)
            for batch in batches:
                y_hat = np.dot(self.x_train[batch], self.beta_mini) + self.beta0_mini
                dl_dbeta0 = (-1 / self.n_sample) * np.sum(self.y_train[batch] - y_hat)
                dl_dbeta = (-1 / self.n_sample) * np.dot(np.transpose(self.x_train[batch]), (y[batch] - y_hat))

                self.beta0_mini = self.beta0_mini - self.l_rate * dl_dbeta0
                self.beta_mini = self.beta_mini - self.l_rate * dl_dbeta

            print(self.beta_mini)
            i += 1


if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=2, n_informative=2, n_targets=1, noise=20, random_state=13)
    lr = LinearRegression()
    lr.batch_gradient_descent(X, y, 0.1, 100)
    lr.stochastic_gradient_descent()
    lr.mini_batch_gradient_descent(10)