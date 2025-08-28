import os, time, json, math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------
# 1) Model builders (as in paper)
# ---------------------------
def build_FNN_BC(input_dim=52):
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])
    return model

def build_FNN_MC(input_dim=52, num_classes=4):
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(210, activation='tanh'),
        layers.Dense(150, activation='tanh'),
        layers.Dropout(0.4),
        layers.Dense(90, activation='tanh'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model

def build_LSTM_BC(input_shape=(1, 52)):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(13, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    return model

def build_LSTM_MC(input_shape=(1, 52), num_classes=4):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(200, activation='tanh', return_sequences=True),
        layers.Dropout(0.5),
        layers.LSTM(150, activation='tanh'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model

# ---------------------------
# 2) Helpers (IID split, scaling, aggregation, testing, history)
# ---------------------------
def data_shuffle(X_train, y_train, arc, users, bs):
    """IID shard per round; returns dict[device_name] -> tf.data.Dataset batched"""
    n = X_train.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    parts = np.array_split(idx, users)
    devices = {}
    for i, p in enumerate(parts):
        Xp, yp = X_train[p], y_train[p]
        ds = tf.data.Dataset.from_tensor_slices((Xp, yp)).shuffle(len(p)).batch(bs, drop_remainder=False)
        devices[f"device_{i+1}"] = ds
    return devices

def scale_calculate(local_weights, devices_data, device_name):
    """Placeholder to keep API parity with pseudocode: returns local_weights unchanged.
    (Scaling is applied at the aggregation step via sample counts.)"""
    return local_weights

def scaled_total_weights(scaled_local_weight_list, sample_sizes):
    """FedAvg aggregation: size-weighted average of weights."""
    total = float(sum(sample_sizes))
    new_weights = []
    for layer_weights in zip(*scaled_local_weight_list):
        agg = np.zeros_like(layer_weights[0])
        for w, sz in zip(layer_weights, sample_sizes):
            agg += (sz / total) * w
        new_weights.append(agg)
    return new_weights

def test_model(X_test, y_test, model, arc):
    """Return loss, accuracy, precision, recall, f1 for the given arc."""
    # Adapt shapes for LSTM arcs
    if arc in ("LSTM_BC", "LSTM_MC"):
        X_eval = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    else:
        X_eval = X_test

    # Keras loss/acc (optional)
    loss, acc = model.evaluate(X_eval, y_test, verbose=0)

    # Sklearn metrics (to match pseudo-code variables)
    if arc.endswith("_BC"):
        y_prob = model.predict(X_eval, verbose=0).ravel()
        y_hat = (y_prob >= 0.5).astype(int)
        average = "binary"
    else:
        y_prob = model.predict(X_eval, verbose=0)
        y_hat = np.argmax(y_prob, axis=1)
        average = "macro"

    precision = precision_score(y_test, y_hat, average=average, zero_division=0)
    recall = recall_score(y_test, y_hat, average=average, zero_division=0)
    f1 = f1_score(y_test, y_hat, average=average, zero_division=0)
    return float(loss), float(acc), float(precision), float(recall), float(f1)

def history_load(history_file):
    """Load .history file created by make_federate; return dict of lists for convenience."""
    rounds, times, losses, accs, pres, recs, f1s = [], [], [], [], [], [], []
    if not os.path.exists(history_file):
        return {"round": rounds, "time": times, "loss": losses, "accuracy": accs,
                "precision": pres, "recall": recs, "f1": f1s}
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            # Expected line format: (round) (time) (loss) (acc) (precision) (recall) (f1)
            # e.g. "(12) (0.5321) (0.1234) (0.9876) (0.98) (0.99) (0.985)"
            toks = [t.strip() for t in line.strip().split() if t.strip()]
            nums = [t.replace("(", "").replace(")", "") for t in toks]
            if len(nums) >= 7:
                rounds.append(int(nums[0]))
                times.append(float(nums[1]))
                losses.append(float(nums[2]))
                accs.append(float(nums[3]))
                pres.append(float(nums[4]))
                recs.append(float(nums[5]))
                f1s.append(float(nums[6]))
    return {"round": rounds, "time": times, "loss": losses, "accuracy": accs,
            "precision": pres, "recall": recs, "f1": f1s}

# ---------------------------
# 3) Core function (faithful to the pseudo-code)
# ---------------------------
def make_federate(
    X, y, arc, model_num, users, acc, local_epochs, optimizer, ts, bs,
    results_dir="../results", max_rounds=10_000, verbose=0
):
    """
    Implements Algorithm 1 from the pseudo-code:
    - X: np.ndarray (n_samples, 52)
    - y: np.ndarray (n_samples,) -> binary {0,1} or multiclass {0,1,2,3}
    - arc: "FNN_BC" | "FNN_MC" | "LSTM_BC" | "LSTM_MC"
    - model_num: int or str (used in history filename)
    - users: number of local clients
    - acc: target accuracy threshold to stop
    - local_epochs: epochs per round per client
    - optimizer: Keras optimizer instance (e.g., Adam(...))
    - ts: test size (float, e.g., 0.2)
    - bs: batch size per client
    """
    os.makedirs(results_dir, exist_ok=True)
    # ---- loss, global model, metrics, verbosity
    if arc == "FNN_BC":
        loss = "binary_crossentropy"
        global_model = build_FNN_BC(input_dim=X.shape[1])
        metrics = ["accuracy"]
    elif arc == "FNN_MC":
        loss = "sparse_categorical_crossentropy"
        global_model = build_FNN_MC(input_dim=X.shape[1], num_classes=4)
        metrics = ["accuracy"]
    elif arc == "LSTM_BC":
        loss = "binary_crossentropy"
        global_model = build_LSTM_BC(input_shape=(1, X.shape[1]))
        metrics = ["accuracy"]
    else:  # "LSTM_MC"
        loss = "sparse_categorical_crossentropy"
        global_model = build_LSTM_MC(input_shape=(1, X.shape[1]), num_classes=4)
        metrics = ["accuracy"]

    global_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    history_file = os.path.join(results_dir, f"model_{model_num}.history")
    with open(history_file, "w", encoding="utf-8") as f:
        pass  # create/clear file

    round_count = 0

    while True:
        round_count += 1
        if round_count > max_rounds:
            print(f"[WARN] Reached max_rounds={max_rounds}. Stopping.")
            break

        global_weights = global_model.get_weights()

        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, stratify=y, random_state=None
        )

        # LSTM arcs need 3D shape for EVAL ONLY; local training uses ds batches (handled automatically by Keras)
        if arc in ("LSTM_BC", "LSTM_MC"):
            X_test_eval = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        else:
            X_test_eval = X_test

        # Make devices (IID shuffled this round)
        devices_data = data_shuffle(X_train, y_train, arc, users, bs)
        devices_names = list(devices_data.keys())

        # Local training across devices
        rtime = 0.0
        scaled_local_weight_list = []
        sample_sizes = []

        for device in devices_names:
            start_time = time.time()

            # Build local model matching arc
            if arc == "FNN_BC":
                local_model = build_FNN_BC(input_dim=X.shape[1])
            elif arc == "FNN_MC":
                local_model = build_FNN_MC(input_dim=X.shape[1], num_classes=4)
            elif arc == "LSTM_BC":
                local_model = build_LSTM_BC(input_shape=(1, X.shape[1]))
            else:
                local_model = build_LSTM_MC(input_shape=(1, X.shape[1]), num_classes=4)

            local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            local_model.set_weights(global_weights)

            # If LSTM arc, Keras will consume (batch, 1, 52) if we feed that shape.
            # Our devices_data is tf.data.Dataset with (X, y); X shape is (batch, 52) for FNN.
            # For LSTM we map to add time dimension:
            ds = devices_data[device]
            if arc in ("LSTM_BC", "LSTM_MC"):
                ds = ds.map(lambda x, t: (tf.reshape(x, (tf.shape(x)[0], 1, tf.shape(x)[1])), t))

            # Train local
            local_model.fit(ds, epochs=local_epochs, verbose=verbose)

            end_time = time.time()
            rtime += (end_time - start_time)

            # Collect weights (scaled later by sample sizes)
            scaled_weights = scale_calculate(local_model.get_weights(), devices_data, device)
            scaled_local_weight_list.append(scaled_weights)

            # Count samples seen by this client this round
            n_samples = 0
            for batch in devices_data[device]:
                n_samples += int(tf.shape(batch[1])[0])
            sample_sizes.append(n_samples)

            tf.keras.backend.clear_session()

        # Round timing (avg per user)
        round_time = round((rtime / users), 4)
        if verbose:
            print(f"Round {round_count} | avg_user_time(s): {round_time}")

        # FedAvg aggregation (size-weighted)
        average_weights = scaled_total_weights(scaled_local_weight_list, sample_sizes)
        global_model.set_weights(average_weights)

        # Evaluate global model
        _loss, _accuracy, _precision, _recall, _f1 = test_model(X_test, y_test, global_model, arc)

        # Append to history file (one line per round)
        with open(history_file, "a", encoding="utf-8") as file:
            file.write(f"({round_count}) ({round_time}) ({_loss}) ({_accuracy}) ({_precision}) ({_recall}) ({_f1})\n")

        # Early stop when reaching target accuracy
        if (_accuracy >= acc):
            break

    # Summarize history
    history = history_load(history_file)
    total_rounds = len(history['round'])
    total_time = round(sum(history['time']), 2)
    print(f"Total rounds: {total_rounds}, Total avg-user-time(s): {total_time}")

    return history

opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# 1. Load dataset
df = pd.read_csv("/Users/sohinikar/M.Tech_Dissertation/Obfuscated-MalMem2022.csv")   # your dataset file

# 2. Separate features and labels
X = df.drop("label", axis=1).values    # all features except label column
y = df["label"].values                 # target column

# 3. Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# X: (N, 52), y: (N,)
hist = make_federate(
    X=X, y=y,
    arc="FNN_BC",            # "FNN_BC" | "FNN_MC" | "LSTM_BC" | "LSTM_MC"
    model_num=1,
    users=8,
    acc=0.999,               # target accuracy to stop
    local_epochs=1,
    optimizer=opt,
    ts=0.2,                  # test size
    bs=128,                  # batch size per client
    results_dir="../results",
    max_rounds=10_000,
    verbose=0
)