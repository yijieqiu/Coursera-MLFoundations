from pla import PLA
from pocket import Pocket


def q15(x, y):
    # Question 15: No input shuffling, update step = 1
    print('Question 15: Running PLA on training samples with update step 1 and no shuffling ...')
    print('Number of updates to PLA convergence: {}'.format(PLA.pla(x, y, False, 1)))
    print()


def q16(x, y):
    # Question 16: Shuffled input, update step = 1, 2000 trials
    print('Question 16: Running PLA on training samples with update step 1 and shuffled input ...')
    updates = []
    for i in range(2000):
        updates.append(PLA.pla(x, y, True, 1))
    print('Average number of updates for PLA to converge with update step 1 and shuffled input {}'
          .format(sum(updates) / len(updates)))
    print()


def q17(x, y):
    # Question 17: Shuffled input, update step = 0.5, 2000 trials
    print('Question 17: Running PLA on training samples with update step 0.5 and shuffled input ...')
    updates = []
    for i in range(2000):
        updates.append(PLA.pla(x, y, True, 0.5))
    print('Average number of updates for PLA to converge with update step 0.5 and shuffled input {}'
          .format(sum(updates) / len(updates)))
    print()


def q18(x_train, y_train, x_eval, y_eval):
    # Question 18: Update step 1, maximum 50 updates, 2000 trials
    print('Question 18: Running 2000 trials...')
    error_rates = []
    for i in range(2000):
        w_pocket, w = Pocket.pocket(x_train, y_train, 1, 50)
        err_rate = Pocket.err_rate(Pocket.yhat(x_eval, w_pocket), y_eval)
        error_rates.append(err_rate)
    print('Average OTS error rate of w_pocket obtained after 50 updates: {}'.format(sum(error_rates) / len(error_rates)))
    print()


def q19(x_train, y_train, x_eval, y_eval):
    # Question 19: Update step = 1, maximum 50 updates, calculate error using w instead of w_pocket, 2000 trials
    print('Question 19: Running 2000 trials...')
    error_rates = []
    for i in range(2000):
        w_pocket, w = Pocket.pocket(x_train, y_train, 1, 50)
        # use w_50 to calculate OTS error rate instead
        err_rate = Pocket.err_rate(Pocket.yhat(x_eval, w), y_eval)
        error_rates.append(err_rate)
    print('Average OTS error rate of w: {}'.format(sum(error_rates) / len(error_rates)))
    print()


def q20(x_train, y_train, x_eval, y_eval):
    # Question 18: Update step 1, maximum 50 updates, 2000 trials
    print('Question 20: Running 2000 trials...')
    error_rates = []
    for i in range(2000):
        w_pocket, w = Pocket.pocket(x_train, y_train, 1, 100)
        err_rate = Pocket.err_rate(Pocket.yhat(x_eval, w_pocket), y_eval)
        error_rates.append(err_rate)
    print('Average OTS error rate of w_pocket obtained after 100 updates: {}'.format(sum(error_rates) / len(error_rates)))
    print()


if __name__ == '__main__':
    x, y = PLA.load_samples('hw1_15_train.dat')
    x_train, y_train = PLA.load_samples('hw1_18_train.dat')
    x_eval, y_eval = PLA.load_samples('hw1_18_test.dat')

    q15(x, y)
    q16(x, y)
    q17(x, y)

    q18(x_train, y_train, x_eval, y_eval)
    q19(x_train, y_train, x_eval, y_eval)
    q20(x_train, y_train, x_eval, y_eval)
