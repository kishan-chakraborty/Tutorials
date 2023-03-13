"""
    To implement simple linear regression using OLS method to get a closed form solution
"""
import numpy as np
from sklearn.datasets import load_diabetes

# pylint: disable = W0201


class SimpleLinearRegression():
    """
        Implementing Multiple Linear Regression from scratch.
    """
    def fit(self, train_x, train_y):
        """
            self.x_train = Independent Variable (Train)
            self.y_train = Dependent Variable (train)
            self.n_sample = No. of training sample
            self.n_features = 1
            Fit the linear regression model to calculate the slope and intercept
        """
        self.x_train = train_x
        self.y_train = train_y
        self.n_sample = len(self.x_train)

        self.beta = np.linalg.inv(np.transpose(self.x_train).dot(self.x_train)).dot(np.transpose(self.x_train).dot(self.y_train))

    def predict(self, x_test):
        """
            Predict value based on created model
        """
        self.val_predicted = x_test * self.beta
        return self.val_predicted


if __name__ == "__main__":
    x_train, y_train = load_diabetes(return_X_y=True)
    x_train = np.insert(x_train, 0, 1, axis=1)  # insert a column of 1s to account for the bias term
    lr = SimpleLinearRegression()
    lr.fit(x_train, y_train)
