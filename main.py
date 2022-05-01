import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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

# show_images(x_test)
clf = MLPClassifier(solver='adam', max_iter=150, activation='logistic', alpha=1e-4, hidden_layer_sizes=(40,), random_state=1)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(score)
