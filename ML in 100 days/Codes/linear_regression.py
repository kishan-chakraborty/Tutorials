"""
    To implement simple linear regression using OLS method to get a closed form solution
"""
import numpy as np
import pandas as pd

# pylint: disable = W0201


class SimpleLinearRegression():
    """
        Implementing Simple Linear Regression from scratch.
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

        beta1_num = np.sum((self.x_train - np.mean(self.x_train)) * (self.y_train - np.mean(self.y_train)))
        beta1_den = np.sum((self.x_train - np.mean(self.x_train))**2)
        self.beta1 = beta1_num / beta1_den
        self.beta0 = np.mean(self.y_train) - self.beta1 * np.mean(self.x_train)

    def predict(self, x_test):
        """
            Predict value based on created model
        """
        self.val_predicted = x_test * self.beta1 + self.beta0
        return self.val_predicted


if __name__ == "__main__":
    df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/100-days-of-machine-learning/main/day48-simple-linear-regression/placement.csv")
    x_train = df['cgpa'].values.reshape(-1, 1)
    y_train = df['package'].values.reshape(-1, 1)
    lr = SimpleLinearRegression()
    lr.fit(x_train, y_train)
