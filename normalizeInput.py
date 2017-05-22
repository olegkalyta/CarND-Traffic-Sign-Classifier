import numpy as np
import pickle


def normalizeImage(image):
    image = image.astype('float')
    for rowIndex in list(np.arange(32)):
        for colIndex in list(np.arange(32)):
            for i in range(3):
                image[rowIndex][colIndex][i] = (image[rowIndex][colIndex][i] - 128) / 128
    return image


with open('data/train.p', mode='rb') as f:
    train = pickle.load(f)
with open('data/valid.p', mode='rb') as f:
    valid = pickle.load(f)
with open('data/test.p', mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)

X_train_optimized = []
for index in list(np.arange(n_train)):
    X_train_optimized.append(normalizeImage(X_train[index]))

X_valid_optimized = []
for index in list(np.arange(n_validation)):
    X_valid_optimized.append(normalizeImage(X_valid[index]))

X_test_optimized = []
for index in list(np.arange(n_test)):
    X_test_optimized.append(normalizeImage(X_test[index]))

print('normalization finished')

with open("data/normalized/x_train_optimized.txt", "wb") as fp:
    pickle.dump(X_train_optimized, fp)

with open("data/normalized/x_valid_optimized.txt", "wb") as fp:
    pickle.dump(X_valid_optimized, fp)

with open("data/normalized/x_test_optimized.txt", "wb") as fp:
    pickle.dump(X_test_optimized, fp)

print('saved')

