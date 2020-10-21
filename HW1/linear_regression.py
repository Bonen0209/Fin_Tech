import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Linear_Regression(object):
    def __init__(self,
                 data_dir,
                 category,
                 output_dir
                ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.category = category

        self.X_train, self.Y_train = self._load_data(mode='train')
        self.X_test, self.Y_test = self._load_data(mode='test')

        self.X_train_with_1 = np.hstack((np.ones((self.X_train.shape[0], 1)), self.X_train))
        self.X_test_with_1 = np.hstack((np.ones((self.X_test.shape[0], 1)), self.X_test))

        self.dim = self.X_train.shape[1]

        self.w = {}
        self.w['w_closed'] = self._closed_form(mode='SVD')
        self.w['w_closed_with_reg'] = self._closed_form(alpha=1)
        self.w['w_closed_with_reg_and_bias'] = self._closed_form(alpha=1, bias=True)
        self.w['w_bayesian'] = self._bayesian(alpha=1)

        self.predicts = {}
        for key in self.w:
            if self.dim == self.w[key].shape[0]:
                self.predicts[key] = self.X_test @ self.w[key]
            else:
                self.predicts[key] = self.X_test_with_1 @ self.w[key]

        self.rmses = {}
        for key in self.w:
            self.rmses[key] = self._rmse(self.predicts[key], self.Y_test)

        if self.category == 'student':
            self._plot_line_chart_predict()
        elif self.category == 'census':
            self._plot_bar_chart_predict()
        self._print_predict()

    def _load_data(self, mode):
        X = np.load(f'{self.data_dir}{self.category}_{mode}_x.npy')
        Y = np.load(f'{self.data_dir}{self.category}_{mode}_y.npy')

        return X, Y

    def _closed_form(self, alpha=0, bias=False, mode='INV'):
        if bias == False:
            X_train = self.X_train
        else:
            X_train = self.X_train_with_1

        dim = X_train.shape[1]
        Y_train = self.Y_train

        INV = X_train.T @ X_train + (alpha / 2) * np.identity(dim)
        INV = self._inv(INV, mode=mode)

        return INV @ X_train.T @ Y_train

    def _bayesian(self, alpha):
        X_train = self.X_train_with_1
        Y_train = self.Y_train
        dim = X_train.shape[1]

        COV = X_train.T @ X_train + self._inv(1/alpha * np.identity(dim), mode='INV')
        COV = self._inv(COV, mode='INV')

        return COV @ (X_train.T @ Y_train + self._inv((1/alpha * np.identity(dim)), mode='INV') @ np.zeros(dim))
     
    def _plot_line_chart_predict(self):

        plt.plot(range(self.X_test.shape[0]), self.Y_test, label=f'Ground Truth')

        for key in self.predicts:
            plt.plot(range(self.predicts[key].shape[0]), self.predicts[key], label=f'RMSE: {np.around(self.rmses[key], 2)} W: {key}')

        plt.legend(loc='lower right')
        plt.xlabel('Sample Index')
        plt.ylabel('Values')
        plt.savefig(f'{self.output_dir}{self.category}_line_chart.png')
        plt.close()

    def _plot_bar_chart_predict(self):
        hist, _ = np.histogram(self.Y_test)
        zeros = [hist[0]]
        ones = [hist[-1]]

        if self.category == 'census':
            for key in self.predicts:
                self.predicts[key] = np.vectorize(self._sign)(self.predicts[key], threshold=0.4)
                hist, _ = np.histogram(self.predicts[key])

                zeros.append(hist[0])
                ones.append(hist[-1])

        ind = np.arange(len(self.predicts) + 1)
        plt.bar(ind, zeros, label='Zero', bottom=ones)
        plt.bar(ind, ones, label='Ones')

        plt.legend(loc='upper right')
        plt.xticks(ind, ('Gound', 'Closed', 'Reg', 'Bias', 'Bayesian'))
        plt.ylabel('Counts')
        plt.savefig(f'{self.output_dir}{self.category}_bar_chart.png')
        plt.close()

    def _print_predict(self):

        for key in self.w:
            print(f'W: {key} RMSE: {np.around(self.rmses[key], 2)}')

    def predict(self, X, key='w_bayesian'):
        return X @ self.w[key]

    @staticmethod
    def _rmse(p, t):
        return np.sqrt(((t - p) ** 2).mean())

    @staticmethod
    def _inv(A, mode):
        if mode == 'SVD':
            U, S, V = np.linalg.svd(A)
            INV = V.T @ np.linalg.inv(np.diag(S)) @ U.T
        elif mode == 'INV':
            INV = np.linalg.inv(A)
        elif mode == 'PINV':
            INV = np.linalg.pinv(A)
            
        return INV

    @staticmethod
    def _sign(x, threshold=0.5):
        if x > threshold:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def _theta(s):
        return 1 / (1 + np.exp(-s))
