import numpy as np
from pla import PLA


class Pocket:

    @staticmethod
    def yhat(x, w):
        """
        Get predicted labels under the current hypothesis
        :param w: Hypothesis to be evaluated
        :param x: Feature matrix
        :return: List of predicted labels
        """
        return list(map(lambda x_i: PLA.sign(x_i, w), x))

    @staticmethod
    def err_rate(yhat, y):
        err = 0.0
        for i in range(len(yhat)):
            if yhat[i] != y[i, 0]:
                err += 1
        return err / len(yhat)

    @staticmethod
    def pocket(x, y, step, max_update):
        """
        Use pocket PLA to learn the best hypothesis for labeling target variable
        :param x: Training feature matrix
        :param y: Training target label matrix
        :param step: PLA weight update step size
        :param max_update: Maximum number of weight updates
        :return: Weight learned upon PLA convergence (best hypothesis), or upon reaching update limit
        """
        # initialize w0 and w_pocket
        w = np.zeros((PLA.dimension, 1))
        w_pocket = np.zeros(w.shape)

        yhat = Pocket.yhat(x, w)
        curr_err = Pocket.err_rate(yhat, y)
        for update in range(max_update):
            err_idx = np.where(yhat != y)[0]
            if len(err_idx) == 0:
                break

            # randomly choose a sample where the current weight makes a mistake
            i = err_idx[np.random.permutation(len(err_idx))[0]]
            # adjust weight vector based on the chosen sample
            w += step * y[i, 0] * np.matrix(x[i]).T

            # get errors after weight update
            yhat = Pocket.yhat(x, w)
            new_err = Pocket.err_rate(yhat, y)
            if new_err < curr_err:
                w_pocket = w.copy()
                curr_err = new_err

        return w_pocket, w


# for local testing only
if __name__ == '__main__':
    x_train, y_train = PLA.load_samples('hw1_18_train.dat')
    x_eval, y_eval = PLA.load_samples('hw1_18_test.dat')

    w_pocket, w = Pocket.pocket(x_train, y_train, 1, 50)
    err_rate = Pocket.err_rate(Pocket.yhat(x_eval, w_pocket), y_eval)
    print('OTS error rate for w_pocket: {}'.format(err_rate))



