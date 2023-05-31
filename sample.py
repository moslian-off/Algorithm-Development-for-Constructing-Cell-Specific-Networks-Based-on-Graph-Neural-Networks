import numpy as np


def num_sample(length, b):
    idx = np.arange(length)
    np.random.shuffle(idx)
    A = idx[:int(0.9 * length)]
    B = idx[int(0.9 * length):]
    return A + b, B + b


def sample(datas, values):
    train_idx = []
    test_idx = []
    c = 0
    num_type = values.shape[0]
    for i in range(num_type):
        train, test = num_sample(values['cell_type'][i], c)
        train_idx += train.tolist()
        test_idx += test.tolist()
        c = c + values['cell_type'][i]
    train_idx = np.random.permutation(np.array(train_idx))
    test_idx = np.random.permutation(np.array(test_idx))
    trainset = [datas[i] for i in train_idx]
    testset = [datas[i] for i in test_idx]
    return trainset, testset
