import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

class IncorrectCounts:
    zero = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    seven = 0
    eight = 0
    nine = 0

    def increment(self, number):
        if number == 0:
            self.zero += 1
        elif number == 1:
            self.one += 1
        elif number == 2:
            self.two += 1
        elif number == 3:
            self.three += 1
        elif number == 4:
            self.four += 1
        elif number == 5:
            self.five += 1
        elif number == 6:
            self.six += 1
        elif number == 7:
            self.seven += 1
        elif number == 8:
            self.eight += 1
        elif number == 9:
            self.nine += 1

    def print_results(self):
        print("Counts of Incorrect Digits:")
        print("ones: " + str(self.one))
        print("twos: " + str(self.two))
        print("threes: " + str(self.three))
        print("fours: " + str(self.four))
        print("fives: " + str(self.five))
        print("sixes: " + str(self.six))
        print("sevens: " + str(self.seven))
        print("eights: " + str(self.eight))
        print("nines: " + str(self.nine))

def incorrect_counts(y_testing, y_prediction):
    incorrect = IncorrectCounts()
    indices = [i for i in range(len(y_testing)) if y_testing[i] != y_prediction[i]]
    for index in indices:
        incorrect.increment(y_testing[index])
    incorrect.print_results()

def show_images(images):
    for img in images:
        B = np.reshape(img, (28, 28))
        plt.imshow(B)
        plt.show()

def load_mnist_data(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = path + '%s-labels.idx1-ubyte' % kind
    images_path = path + '%s-images.idx3-ubyte' % kind

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

x_train, y_train = load_mnist_data("./Data/", "train")
x_test, y_test = load_mnist_data("./Data/", "t10k")

# Best so far is 784 random state of 42 at 96.99
clf = MLPClassifier(solver='adam', max_iter=150, activation='logistic', alpha=1e-4,
                    hidden_layer_sizes=(784, ), random_state=42)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(score)

# Find incorrect counts
incorrect_counts(y_test, y_pred)

train_sizes, train_scores, test_scores = learning_curve(clf, x_train, y_train,
                                                        train_sizes=[300, 600, 2000, 10000, 25000, 45000], cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
axes = None
if axes is None:
    _, axes = plt.subplots(1, 2, figsize=(20, 5))
# Plot learning curve
axes[0].fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="r",
)
axes[0].fill_between(
    train_sizes,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="g",
)
axes[0].plot(
    train_sizes, train_scores_mean, "o-", color="r", label="Training score"
)
axes[0].plot(
    train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
)
axes[0].legend(loc="best")

plt.show()
