import numpy as np


def stack_train_test_orientation(train, test) :
    train_data, train_labels = train
    test_data, test_labels = test
    train_data = np.vstack(train_data)
    train_labels = np.hstack(train_labels)
    test_data = np.vstack(test_data)
    test_labels = np.hstack(test_labels)
    return train_data, train_labels, test_data, test_labels
