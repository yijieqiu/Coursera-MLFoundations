import os
import numpy as np
import random


class PLA:
    """
    Base class for HW1 questions 15-17. Implements PLA and allows for shuffling of input
    order and configurable number of iterations
    """

    # Dimension shared by feature vector X and weight vector W. 1 (bias constant) + 4 (feature count)
    dimension = 5

    @staticmethod
    def load_samples(file_name):
        """
        Read and parse input data from current directory, one line at a time.
        Input format for line i: x_i0 x_i1 x_i2 y_i

        :param file_name: Name of file containing samples, from the same directory
        :return Two matrices X, Y, containing features and labels for training samples, respectively
        """
        pwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        print('Reading training input file {} under path {}'.format(file_name, pwd))
        with open(os.path.join(pwd, file_name)) as f:
            samples = f.readlines()
            rows = len(samples)

            # X matrix is of dimension rows x 5, column 0 being the model constant, other four being feature values
            x = np.zeros((rows, PLA.dimension))
            # X matrix is of dimension rows x 1, containing all training labels
            y = np.zeros((rows, 1))
            for i in range(rows):
                sample = samples[i].strip().split()
                x[i, 0] = 1.0
                x[i, 1] = np.float(sample[0])
                x[i, 2] = np.float(sample[1])
                x[i, 3] = np.float(sample[2])
                x[i, 4] = np.float(sample[3])

                y[i, 0] = np.int(sample[4])

        return x, y

    @staticmethod
    def get_index(n, shuffle):
        """
        Returns indices at which training samples will be examined
        :param n: Size of indices
        :param shuffle: Whether indices should be shuffled for random access
        """
        index = range(n)
        if shuffle:
            index = np.random.permutation(index)
        return index

    @staticmethod
    def sign(x, w):
        """
        Provide target label (1 or -1) using feature vector and current weights
        :param x: Feature vector
        :param w: Weight
        :return: Target label
        """
        if np.dot(x, w)[0] > 0:
            return 1
        else:
            return -1

    @staticmethod
    def pla(x, y, shuffle, step):
        """
        Use PLA to learn the best hypothesis for labeling target variable
        :param x: Data frame of feature vectors x0,x1...,xn from training sample
        :param y: Data frame of labels from training sample
        :param shuffle: Whether input samples should be shuffled
        :param step: Step size for PLA weight update
        :return: Number of weight updates to convergence
        :raise ArithmeticError if PLA fails to converge within 500 updates
        """
        # initialize w0
        w = np.zeros((PLA.dimension, 1))
        # number of weight updates
        updates = 0
        # whether PLA has converged (no labeling error for current w)
        converged = False

        index = PLA.get_index(len(x), shuffle)
        while not converged:
            # arbitrary stopping criteria in case PLA fails to converge
            if updates > 500:
                raise ArithmeticError('PLA failed to converge after {} weight updates!'.format(updates))
            for i in index:
                if PLA.sign(x[i], w) == y[i, 0]:
                    converged = True
                else:
                    converged = False
                    # make correction to w
                    w += step * y[i, 0] * np.matrix(x[i]).T
                    updates += 1

            if converged:
                break

        return updates


# for local testing only
if __name__ == '__main__':
    x, y = PLA.load_samples('hw1_15_train.dat')
    PLA.pla(x, y, False, 1)
