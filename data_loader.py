import numpy as np

def load_images(file):
    with open(file, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28*28) / 255.0

def load_labels(file):
    with open(file, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data

def get_data(batch_size=128):
    X_train = load_images("train-images.idx3-ubyte")
    y_train = load_labels("train-labels.idx1-ubyte")

    X_test = load_images("t10k-images.idx3-ubyte")
    y_test = load_labels("t10k-labels.idx1-ubyte")
    

    def create_batches(X, y):
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

    train_loader = list(create_batches(X_train, y_train))
    test_loader = list(create_batches(X_test, y_test))

    return train_loader, test_loader