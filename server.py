from __future__ import annotations

import flwr as fl


def main() -> None:
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=3,
        min_available_clients=3,
        on_fit_config_fn=lambda round_id: {"lr": 1e-3, "round": round_id},
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
