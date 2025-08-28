#imports

import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg


# Stores training and evaluation accuracy and loss for each round
round_metrics = {
    "fit_accuracy": [],
    "eval_accuracy": [],
    "fit_loss": [],
    "eval_loss": [],
}

#customizing fedavg as by default in fedavg on 2 clients can be used
class CustomFedAvg(FedAvg):
    # Called after every training round, retrieves and logs accuracy and loss from aggregated metrics
    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated:
            _, metrics = aggregated
            acc = metrics.get("accuracy", 0.0)
            loss = metrics.get("loss", 0.0)
            round_metrics["fit_accuracy"].append((rnd, acc))
            round_metrics["fit_loss"].append((rnd, loss))
        return aggregated
    
    # same for evaluate
    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        if aggregated:
            _, metrics = aggregated  
            acc = metrics.get("accuracy", 0.0)
            loss = metrics.get("loss", 0.0)
            round_metrics["eval_accuracy"].append((rnd, acc))
            round_metrics["eval_loss"].append((rnd, loss))
        return aggregated

# These calculate weighted mean values based on the number of examples from each client:

# More examples = more influence in the average

# Used to aggregate client metrics during training and evaluation

def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(
        metrics_dict.get("accuracy", 0) * num_examples
        for num_examples, metrics_dict in metrics
    ) / total_examples

    return {"accuracy": weighted_accuracy}


def weighted_loss(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_loss = sum(
        metrics_dict.get("loss", 0) * num_examples
        for num_examples, metrics_dict in metrics
    ) / total_examples

    return {"loss": weighted_loss}


def print_summary():
    print("\n=========== Federated Learning Summary ===========")
    print(f"Total Rounds: {len(round_metrics['fit_accuracy'])}\n")

    print("Training Metrics (Fit):")
    print("{:<8} {:<10} {:<10}".format("Round", "Accuracy", "Loss"))
    for (r_acc, acc), (_, loss) in zip(round_metrics["fit_accuracy"], round_metrics["fit_loss"]):
        print(f"{r_acc:<8} {acc:<10.4f} {loss:<10.4f}")

    print("\nEvaluation Metrics:")
    print("{:<8} {:<10} {:<10}".format("Round", "Accuracy", "Loss"))
    for (r_acc, acc), (_, loss) in zip(round_metrics["eval_accuracy"], round_metrics["eval_loss"]):
        print(f"{r_acc:<8} {acc:<10.4f} {loss:<10.4f}")
    print("==================================================")


# Server setup
if __name__ == "__main__":
    print("[Server] Starting Federated Learning server...")

    strategy = CustomFedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        fit_metrics_aggregation_fn=lambda m: {
            **weighted_average(m),
            **weighted_loss(m),
        },
        evaluate_metrics_aggregation_fn=lambda m: {
            **weighted_average(m),
            **weighted_loss(m),
        },
    )

    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    print_summary()
    print("[Server] Training complete.")
