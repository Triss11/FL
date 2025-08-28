import sys
import flwr as fl
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from collections import Counter

# Load and preprocess data
X_scaled = pd.read_csv("/Users/sohinikar/M.Tech_Dissertation/X_test_scaled.csv")
y = pd.read_csv("/Users/sohinikar/M.Tech_Dissertation/y_test.csv")

# Simulate a client with just a slice of the data
def get_client_data(client_id: int, total_clients: int = 3):
    size = len(X_scaled) // total_clients
    start = client_id * size
    end = start + size
    X_local = X_scaled[start:end]
    y_local = y[start:end]
    # label_counts = Counter(y_local)
    # print(f"[Client {client_id}] Data label distribution: {label_counts}")
    return np.array(X_local, dtype=np.float32), np.array(y_local, dtype=np.float32)

def sample(X, y, benign_idx, malware_idx, b_ratio, m_ratio, size=1000):
        b_count = int(b_ratio * size)
        m_count = int(m_ratio * size)
        indices = np.concatenate([benign_idx[:b_count], malware_idx[:m_count]])
        np.random.shuffle(indices)
        return X[indices], y[indices]

# creating custom splits if 50-50, 70-30 and 30-70
def create_custom_splits(X, y):
    benign_idx = np.where(y == 0)[0]
    malware_idx = np.where(y == 1)[0]
    np.random.shuffle(benign_idx)
    np.random.shuffle(malware_idx)

    split_50_50 = sample(X, y, benign_idx, malware_idx, 0.5, 0.5)
    split_70_benign = sample(X, y, benign_idx, malware_idx, 0.7, 0.3)
    split_70_malware = sample(X, y, benign_idx, malware_idx, 0.3, 0.7)

    return split_50_50, split_70_benign, split_70_malware

class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = tf.keras.models.load_model("cic_malmem_model.h5")
        self.X, self.y = get_client_data(client_id)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = self.model.fit(self.X, self.y, epochs=1, batch_size=32, verbose=0)
        acc = float(history.history["accuracy"][0])
        loss = float(history.history["loss"][0])
        print(f"\n[Client {self.client_id}] Training -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return self.model.get_weights(), len(self.X), {"accuracy": acc}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Standard evaluation on local slice
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)

        # Custom splits
        (X_50_50, y_50_50), (X_70_b, y_70_b), (X_70_m, y_70_m) = create_custom_splits(self.X, self.y)

        acc_50 = self.model.evaluate(X_50_50, y_50_50, verbose=0)[1]
        acc_70_b = self.model.evaluate(X_70_b, y_70_b, verbose=0)[1]
        acc_70_m = self.model.evaluate(X_70_m, y_70_m, verbose=0)[1]

        print(f"[Client {self.client_id}] Eval -> Loss: {loss:.4f}, Acc: {acc:.4f}")
        print(f"[Client {self.client_id}] 50-50 Acc: {acc_50:.4f}, 70% Benign Acc: {acc_70_b:.4f}, 70% Malware Acc: {acc_70_m:.4f}")

        return loss, len(self.X), {
            "accuracy": acc,
            "acc_50_50": acc_50,
            "acc_70_benign": acc_70_b,
            "acc_70_malware": acc_70_m,
        }

# Start the client
client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FLClient(client_id))
