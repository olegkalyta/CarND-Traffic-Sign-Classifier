import pickle


def init():
    training_file = 'data/train.p'
    validation_file = 'data/valid.p'
    # testing_file = 'data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)

    y_train = train['labels']
    y_valid = valid['labels']

    with open('data/normalized/x_train_optimized.txt', mode='rb') as f:
        X_train_optimized = pickle.load(f)
    with open('data/normalized/x_valid_optimized.txt', mode='rb') as f:
        X_valid_optimized = pickle.load(f)

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)

    y_train = train['labels']
    y_valid = valid['labels']

    with open('data/normalized/x_train_optimized.txt', mode='rb') as f:
        X_train_optimized = pickle.load(f)
    with open('data/normalized/x_valid_optimized.txt', mode='rb') as f:
        X_valid_optimized = pickle.load(f)

    return [y_train, y_valid, X_train_optimized, X_valid_optimized]