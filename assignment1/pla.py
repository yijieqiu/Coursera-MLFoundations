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
    # Static training input file name. Assuming that training data is always available under the current directory
    file_name = 'hw1_15_train.dat'

    @staticmethod
    def get_training_samples():
        """
        Read training data from current directory, one line at a time.
        Input format for line i: x_i0 x_i1 x_i2 y_i

        :return Two matrices X, Y, containing features and labels for training samples, respectively
        """
        pwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        print('Reading training input file {} under path {}'.format(PLA.file_name, pwd))
        with open(os.path.join(pwd, PLA.file_name)) as f:
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
        :return: final weight and number of iterations to convergence
        :raise ArithmeticError if PLA fails to converge within 500 iterations
        """
        # initialize w0
        w = np.zeros((PLA.dimension, 1))
        # number of iterations
        iterations = 0
        # whether PLA has converged (no labeling error for current w)
        converged = False

        # explicitly supply an array of indices, to accommodate the potential need for input shuffling
        index = range(len(x))
        if shuffle:
            # Can't use random.shuffle() because range object does not support assignment
            index = random.sample(index, len(x))
        while not converged:
            # arbitrary stopping criteria in case PLA fails to converge
            if iterations > 500:
                raise ArithmeticError('PLA failed to converge after {} iterations!'.format(iterations))
            for i in index:
                if PLA.sign(x[i], w) == y[i, 0]:
                    converged = True
                else:
                    converged = False
                    # make correction to w
                    w = w + step * y[i, 0] * np.matrix(x[i]).T
                    iterations += 1

            if converged:
                break

        return iterations


# For local testing only
if __name__ == '__main__':
    # Training data can be read once and reused across invocations
    x, y = PLA.get_training_samples()

    # Question 15: No input shuffling, update step = 1
    print('Question 15: Running PLA on training samples with update step 1 and no shuffling ...')
    print('Number of iterations to PLA convergence: {}'.format(PLA.pla(x, y, False, 1)))
    print()

    # Question 16: Shuffled input, update step = 1, 2000 trials
    print('Question 16: Running PLA on training samples with update step 1 and shuffled input ...')
    iterations = []
    for i in range(2000):
        iterations.append(PLA.pla(x, y, True, 1))
    print('Average number of iterations for PLA to converge with update step 1 and shuffled input {}'
          .format(sum(iterations) / len(iterations)))
    print()

    # Question 17: Shuffled input, update step = 0.5, 2000 trials
    print('Question 17: Running PLA on training samples with update step 0.5 and shuffled input ...')
    iterations = []
    for i in range(2000):
        iterations.append(PLA.pla(x, y, True, 0.5))
    print('Average number of iterations for PLA to converge with update step 0.5 and shuffled input {}'
          .format(sum(iterations) / len(iterations)))
    print()






