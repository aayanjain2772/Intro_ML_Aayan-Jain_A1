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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For prettier confusion matrix plots (optional)
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# 1. Preprocessing Functions
# -----------------------------
def preprocess_data(X_train, X_test, method='none'):
    """
    Preprocess the data using:
    - 'none': leave data unchanged.
    - 'normalize': apply min-max scaling to [0, 1] using training data parameters.
    - 'standardize': apply z-score normalization using training data mean and std.
    """
    if method == 'normalize':
        X_train_proc = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
        X_test_proc = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
    elif method == 'standardize':
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train_proc = (X_train - mean) / std
        X_test_proc = (X_test - mean) / std
    else:  # method == 'none'
        X_train_proc, X_test_proc = X_train, X_test
    return X_train_proc, X_test_proc

def one_hot_encode(y, num_classes):
    """Convert label vector y to one-hot encoded matrix."""
    return np.eye(num_classes)[y]

# -----------------------------
# 2. Utility Functions
# -----------------------------
def softmax(z):
    """
    Compute the softmax for each row of the input z.
    Subtracting the row-wise maximum ensures numerical stability.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """
    Compute the average categorical cross-entropy loss.
    y_true: one-hot encoded true labels.
    y_pred: predicted probabilities.
    """
    epsilon = 1e-15  # small constant for numerical stability
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# -----------------------------
# 3. Training Functions
# -----------------------------
def train_gd(X, y, learning_rate=0.1, epochs=1000):
    """
    Train logistic regression using Batch Gradient Descent.
    X: training features, shape (m, n)
    y: one-hot encoded labels, shape (m, num_classes)
    Returns the learned weights, bias, and loss history.
    """
    m, n = X.shape
    num_classes = y.shape[1]
    
    # Initialize weights and bias
    W = np.zeros((n, num_classes))
    b = np.zeros((1, num_classes))
    
    loss_history = []
    
    for epoch in range(epochs):
        # Compute logits and predictions
        logits = np.dot(X, W) + b
        y_pred = softmax(logits)
        
        # Compute loss and store
        loss = cross_entropy_loss(y, y_pred)
        loss_history.append(loss)
        
        # Compute gradients
        grad_logits = (y_pred - y) / m
        grad_W = np.dot(X.T, grad_logits)
        grad_b = np.sum(grad_logits, axis=0, keepdims=True)
        
        # Update parameters
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b
        
        if (epoch+1) % 100 == 0:
            print(f"[GD] Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    return W, b, loss_history

def train_sgd(X, y, learning_rate=0.01, epochs=1000):
    """
    Train logistic regression using Stochastic Gradient Descent.
    X: training features, shape (m, n)
    y: one-hot encoded labels, shape (m, num_classes)
    Returns the learned weights, bias, and loss history.
    """
    m, n = X.shape
    num_classes = y.shape[1]
    
    # Initialize weights and bias
    W = np.zeros((n, num_classes))
    b = np.zeros((1, num_classes))
    
    loss_history = []
    
    for epoch in range(epochs):
        # Shuffle the training data at each epoch
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process one sample at a time
        for i in range(m):
            xi = X_shuffled[i:i+1]  # shape (1, n)
            yi = y_shuffled[i:i+1]  # shape (1, num_classes)
            logits = np.dot(xi, W) + b
            y_pred = softmax(logits)
            # Compute gradients based on the sample
            grad_logits = (y_pred - yi)
            grad_W = np.dot(xi.T, grad_logits)
            grad_b = grad_logits
            
            # Update parameters
            W -= learning_rate * grad_W
            b -= learning_rate * grad_b
        
        # Compute loss over the entire training set at the end of the epoch
        logits_full = np.dot(X, W) + b
        y_pred_full = softmax(logits_full)
        epoch_loss = cross_entropy_loss(y, y_pred_full)
        loss_history.append(epoch_loss)
        
        if (epoch+1) % 100 == 0:
            print(f"[SGD] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return W, b, loss_history

# -----------------------------
# 4. Main Workflow
# -----------------------------
# (a) Load the Dataset
iris = load_iris()
X = iris.data    # shape: (150, 4)
y = iris.target  # shape: (150,)
num_classes = len(np.unique(y))
y_encoded = one_hot_encode(y, num_classes)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique classes:", np.unique(y))

# (b) Preprocess the Data
# Split into training (60%) and test (40%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, random_state=42)

# Choose preprocessing method: 'none', 'normalize', or 'standardize'
preprocess_method = 'standardize'  # Change as needed
X_train_proc, X_test_proc = preprocess_data(X_train, X_test, method=preprocess_method)

# (c) & (d) Implement and Train Multiclass Logistic Regression

# --- Training with Batch Gradient Descent ---
print("\nTraining with Batch Gradient Descent:")
W_gd, b_gd, loss_history_gd = train_gd(X_train_proc, y_train, learning_rate=0.1, epochs=1000)

# Visualize the training loss curve for GD
plt.figure(figsize=(8, 5))
plt.plot(loss_history_gd)
plt.title("Training Loss Curve (Gradient Descent)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Evaluate the GD model on the test set
logits_test_gd = np.dot(X_test_proc, W_gd) + b_gd
y_pred_prob_gd = softmax(logits_test_gd)
y_pred_gd = np.argmax(y_pred_prob_gd, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("\n[GD] Test Accuracy:", accuracy_score(y_test_labels, y_pred_gd))
print("[GD] Classification Report:\n", classification_report(y_test_labels, y_pred_gd))
print("[GD] Confusion Matrix:\n", confusion_matrix(y_test_labels, y_pred_gd))

# --- Training with Stochastic Gradient Descent ---
print("\nTraining with Stochastic Gradient Descent:")
W_sgd, b_sgd, loss_history_sgd = train_sgd(X_train_proc, y_train, learning_rate=0.01, epochs=1000)

# Visualize the training loss curve for SGD
plt.figure(figsize=(8, 5))
plt.plot(loss_history_sgd)
plt.title("Training Loss Curve (Stochastic Gradient Descent)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Evaluate the SGD model on the test set
logits_test_sgd = np.dot(X_test_proc, W_sgd) + b_sgd
y_pred_prob_sgd = softmax(logits_test_sgd)
y_pred_sgd = np.argmax(y_pred_prob_sgd, axis=1)

print("\n[SGD] Test Accuracy:", accuracy_score(y_test_labels, y_pred_sgd))
print("[SGD] Classification Report:\n", classification_report(y_test_labels, y_pred_sgd))
print("[SGD] Confusion Matrix:\n", confusion_matrix(y_test_labels, y_pred_sgd))
