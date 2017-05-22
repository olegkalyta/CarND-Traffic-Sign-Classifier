import pickle


def init():
    with open('data/train.p', mode='rb') as f:
        train = pickle.load(f)
    with open('data/valid.p', mode='rb') as f:
        valid = pickle.load(f)
    with open('data/test.p', mode='rb') as f:
        test = pickle.load(f)

    y_train = train['labels']
    y_valid = valid['labels']
    y_test = test['labels']

    with open('data/normalized/x_train_optimized.txt', mode='rb') as f:
        X_train_optimized = pickle.load(f)
    with open('data/normalized/x_valid_optimized.txt', mode='rb') as f:
        X_valid_optimized = pickle.load(f)
    with open('data/normalized/x_test_optimized.txt', mode='rb') as f:
        X_test_optimized = pickle.load(f)

    return [X_train_optimized, X_valid_optimized, X_test_optimized, y_train, y_valid, y_test]
