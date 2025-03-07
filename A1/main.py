import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns  # Optional

def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    m = y.shape[0]
    y_encoded = np.zeros((m, num_classes))
    y_encoded[np.arange(m), y] = 1
    return y_encoded

def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    epsilon = 1e-15
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
    return loss

class LogisticRegressionMulti:
    def __init__(self, learning_rate=0.01, epochs=1000, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
    
    def fit_gd(self, X, y):
        m, n = X.shape
        num_classes = len(np.unique(y))
        self.W = np.zeros((n, num_classes))
        y_encoded = one_hot_encode(y, num_classes)
        self.loss_history = []
        
        for epoch in range(self.epochs):
            logits = np.dot(X, self.W)
            probs = softmax(logits)
            loss = compute_loss(y_encoded, probs)
            self.loss_history.append(loss)
            
            grad = np.dot(X.T, (probs - y_encoded)) / m
            self.W -= self.learning_rate * grad
            
            if self.verbose and epoch % 100 == 0:
                print(f"[GD] Epoch {epoch}/{self.epochs} - Loss: {loss:.4f}")
        return self
    
    def fit_sgd(self, X, y):
        m, n = X.shape
        num_classes = len(np.unique(y))
        self.W = np.zeros((n, num_classes))
        y_encoded = one_hot_encode(y, num_classes)
        self.loss_history = []
        
        for epoch in range(self.epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            for i in indices:
                xi = X[i:i+1]
                yi = y_encoded[i:i+1]
                logits = np.dot(xi, self.W)
                probs = softmax(logits)
                grad = np.dot(xi.T, (probs - yi))
                self.W -= self.learning_rate * grad
            logits = np.dot(X, self.W)
            probs = softmax(logits)
            loss = compute_loss(y_encoded, probs)
            self.loss_history.append(loss)
            
            if self.verbose and epoch % 100 == 0:
                print(f"[SGD] Epoch {epoch}/{self.epochs} - Loss: {loss:.4f}")
        return self
    
    def predict(self, X):
        logits = np.dot(X, self.W)
        probs = softmax(logits)
        return np.argmax(probs, axis=1)

def main():
    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Unique classes:", np.unique(y))
    
    # Split the dataset (60% train, 40% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    
    # Preprocess the data: choose normalization (or standardization/unprocessed)
    X_train_proc = normalize(X_train)
    X_test_proc  = normalize(X_test)
    # For standardized data, use:
    # X_train_proc = standardize(X_train)
    # X_test_proc  = standardize(X_test)
    
    # Train using Batch Gradient Descent (GD)
    model_gd = LogisticRegressionMulti(learning_rate=0.1, epochs=1000, verbose=True)
    model_gd.fit_gd(X_train_proc, y_train)
    y_pred_gd = model_gd.predict(X_test_proc)
    print("\n[GD] Test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_gd) * 100))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_gd))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gd))
    
    # Plot the loss curve for GD
    plt.figure(figsize=(8, 5))
    plt.plot(model_gd.loss_history, label="GD Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss Curve (Gradient Descent)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Train using Stochastic Gradient Descent (SGD)
    model_sgd = LogisticRegressionMulti(learning_rate=0.01, epochs=1000, verbose=True)
    model_sgd.fit_sgd(X_train_proc, y_train)
    y_pred_sgd = model_sgd.predict(X_test_proc)
    print("\n[SGD] Test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_sgd) * 100))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_sgd))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_sgd))
    
    # Plot the loss curve for SGD
    plt.figure(figsize=(8, 5))
    plt.plot(model_sgd.loss_history, label="SGD Loss", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss Curve (Stochastic Gradient Descent)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
