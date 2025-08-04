import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.datasets import mnist

def _change_one_hot_label(X, num_classes=10):
    return np.eye(num_classes)[X]

def get_data(normalize=True, flatten=True, one_hot_label=False):
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
    if flatten:
        x_train = x_train.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)
    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)
    return (x_train, t_train), (x_test, t_test)

def init_network():
    weight_path = Path.cwd() / 'sample_weight.pkl'
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found at {weight_path.resolve()}")
    with open(weight_path, 'rb') as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)  # 오버플로 대책
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

(x_train, t_train), (x_test, t_test) = get_data(normalize=True, flatten=True, one_hot_label=False)
network = init_network()

batch_size = 100
accuracy_int = 0
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i + batch_size]
    y_batch = predict(network, x_batch)

    p = np.argmax(y_batch, axis=1)
    accuracy_int += np.sum(p == t_test[i:i + batch_size])

print("Accuracy:" + str(accuracy_int / len(x_test)))

