import numpy as np
from model import *
from utils import *
from data_loader import get_data


def one_hot(y, num_classes=10):
    one_hot_encoded = np.zeros((len(y), num_classes))
    one_hot_encoded[np.arange(len(y)), y] = 1
    return one_hot_encoded


def train(train_loader, epochs=30, lr=0.1):
    W1, b1, W2, b2 = initialize_parameters(784, 128, 10)

    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in train_loader:

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)

            # flatten images
            X_batch = X_batch.reshape(X_batch.shape[0], -1)

            # one-hot labels
            y_batch = one_hot(y_batch)

            # forward
            y_pred, cache = forward(X_batch, W1, b1, W2, b2)

            # loss
            loss = compute_loss(y_batch, y_pred)
            total_loss += loss

            # backward
            dW1, db1, dW2, db2 = backward(cache, W2, y_batch)

            # update
            W1, b1, W2, b2 = update_parameters(
                W1, b1, W2, b2,
                dW1, db1, dW2, db2, lr
            )

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return W1, b1, W2, b2


def predict(X, W1, b1, W2, b2):
    y_pred, _ = forward(X, W1, b1, W2, b2)
    return np.argmax(y_pred, axis=1)


def main():
    train_loader, test_loader = get_data()

    # Train
    W1, b1, W2, b2 = train(train_loader)

    # Test
    all_preds = []
    all_labels = []

    for X_test, y_test in test_loader:
        X_test = np.array(X_test)
        X_test = X_test.reshape(X_test.shape[0], -1)

        preds = predict(X_test, W1, b1, W2, b2)

        all_preds.extend(preds)
        all_labels.extend(y_test)

    acc = accuracy(np.array(all_labels), np.array(all_preds))
    print("Final Test Accuracy:", acc)


if __name__ == "__main__":
    main()