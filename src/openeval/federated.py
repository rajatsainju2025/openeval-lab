from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol
import time

from ..core import Task, Dataset, Adapter, Metric


class PrivacyMechanism(Protocol):
    """Protocol for differential privacy mechanisms."""

    def add_noise(self, value: float, sensitivity: float, epsilon: float) -> float:
        """Add noise to a value for privacy."""
        ...

    def get_privacy_budget(self) -> float:
        """Get remaining privacy budget."""
        ...


@dataclass
class FederatedConfig:
    """Configuration for federated evaluation."""

    num_clients: int = 5
    rounds: int = 3
    epsilon: float = 1.0  # Privacy budget
    min_clients_per_round: int = 3
    max_clients_per_round: int = 5
    client_selection_strategy: str = "random"  # random, importance, diversity
    aggregation_method: str = "fedavg"  # fedavg, fedprox, scaffold
    privacy_mechanism: Optional[str] = "gaussian"  # gaussian, laplace, none


@dataclass
class ClientUpdate:
    """Update from a federated client."""

    client_id: str
    local_metrics: Dict[str, float]
    local_dataset_size: int
    model_parameters: Optional[Dict[str, Any]] = None
    privacy_noise: Optional[Dict[str, float]] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class FederatedResult:
    """Result of federated evaluation."""

    global_metrics: Dict[str, float]
    client_updates: List[ClientUpdate]
    privacy_budget_used: float
    convergence_metrics: Dict[str, List[float]]
    round: int
    total_clients: int


class FederatedEvaluator:
    """Federated evaluation coordinator."""

    def __init__(self, config: FederatedConfig):
        self.config = config
        self.clients: Dict[str, FederatedClient] = {}
        self.global_model = None
        self.privacy_budget_used = 0.0
        self.convergence_history: Dict[str, List[float]] = {}

    def register_client(self, client: FederatedClient) -> None:
        """Register a client for federated evaluation."""
        self.clients[client.client_id] = client

    def select_clients(self, round_num: int) -> List[FederatedClient]:
        """Select clients for the current round."""
        available_clients = list(self.clients.values())

        if len(available_clients) <= self.config.max_clients_per_round:
            return available_clients

        if self.config.client_selection_strategy == "random":
            import random

            return random.sample(available_clients, self.config.max_clients_per_round)
        elif self.config.client_selection_strategy == "importance":
            # Select based on dataset size or previous performance
            return sorted(available_clients, key=lambda c: c.dataset_size, reverse=True)[
                : self.config.max_clients_per_round
            ]
        else:
            # Default to random
            import random

            return random.sample(available_clients, self.config.max_clients_per_round)

    def aggregate_updates(self, updates: List[ClientUpdate]) -> Dict[str, float]:
        """Aggregate client updates using specified method."""
        if not updates:
            return {}

        if self.config.aggregation_method == "fedavg":
            return self._fedavg_aggregation(updates)
        elif self.config.aggregation_method == "fedprox":
            return self._fedprox_aggregation(updates)
        else:
            return self._fedavg_aggregation(updates)

    def _fedavg_aggregation(self, updates: List[ClientUpdate]) -> Dict[str, float]:
        """Federated averaging aggregation."""
        if not updates:
            return {}

        # Get all metric names
        metric_names = set()
        for update in updates:
            metric_names.update(update.local_metrics.keys())

        aggregated_metrics = {}
        total_weight = sum(update.local_dataset_size for update in updates)

        for metric_name in metric_names:
            weighted_sum = 0.0
            total_contributing_weight = 0.0

            for update in updates:
                if metric_name in update.local_metrics:
                    weight = update.local_dataset_size / total_weight
                    weighted_sum += update.local_metrics[metric_name] * weight
                    total_contributing_weight += weight

            if total_contributing_weight > 0:
                aggregated_metrics[metric_name] = weighted_sum / total_contributing_weight
            else:
                aggregated_metrics[metric_name] = 0.0

        return aggregated_metrics

    def _fedprox_aggregation(self, updates: List[ClientUpdate]) -> Dict[str, float]:
        """FedProx aggregation with proximal term."""
        # Simplified FedProx - in practice would include proximal regularization
        return self._fedavg_aggregation(updates)

    def add_privacy_noise(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Add privacy-preserving noise to metrics."""
        if self.config.privacy_mechanism == "none":
            return metrics

        noisy_metrics = {}
        sensitivity = 1.0  # Simplified sensitivity assumption

        for metric_name, value in metrics.items():
            if self.config.privacy_mechanism == "gaussian":
                # Gaussian mechanism
                import numpy as np

                noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / 0.01)) / self.config.epsilon
                noise = np.random.normal(0, noise_scale)
            elif self.config.privacy_mechanism == "laplace":
                # Laplace mechanism
                import numpy as np

                noise_scale = sensitivity / self.config.epsilon
                noise = np.random.laplace(0, noise_scale)
            else:
                noise = 0.0

            noisy_metrics[metric_name] = value + noise
            self.privacy_budget_used += self.config.epsilon / len(metrics)

        return noisy_metrics

    def check_convergence(self, current_metrics: Dict[str, float], round_num: int) -> bool:
        """Check if federated learning has converged."""
        if round_num < 2:
            return False

        # Simple convergence check based on metric stability
        for metric_name, values in self.convergence_history.items():
            if len(values) >= 2:
                recent_change = abs(values[-1] - values[-2]) / max(abs(values[-2]), 1e-10)
                if recent_change > 0.01:  # 1% change threshold
                    return False

        return True

    def run_federated_evaluation(
        self, task: Task, global_dataset: Dataset, adapter: Adapter, metrics: List[Metric]
    ) -> List[FederatedResult]:
        """Run federated evaluation for multiple rounds."""
        results = []

        for round_num in range(self.config.rounds):
            # Select clients for this round
            selected_clients = self.select_clients(round_num)

            if len(selected_clients) < self.config.min_clients_per_round:
                break  # Not enough clients

            # Collect client updates
            client_updates = []
            for client in selected_clients:
                update = client.evaluate_locally(task, adapter, metrics)
                client_updates.append(update)

            # Aggregate updates
            global_metrics = self.aggregate_updates(client_updates)

            # Add privacy noise if configured
            if self.config.privacy_mechanism != "none":
                global_metrics = self.add_privacy_noise(global_metrics)

            # Update convergence history
            for metric_name, value in global_metrics.items():
                if metric_name not in self.convergence_history:
                    self.convergence_history[metric_name] = []
                self.convergence_history[metric_name].append(value)

            # Create result
            result = FederatedResult(
                global_metrics=global_metrics,
                client_updates=client_updates,
                privacy_budget_used=self.privacy_budget_used,
                convergence_metrics=self.convergence_history.copy(),
                round=round_num,
                total_clients=len(selected_clients),
            )
            results.append(result)

            # Check convergence
            if self.check_convergence(global_metrics, round_num):
                break

        return results


@dataclass
class FederatedClient:
    """A client in the federated evaluation network."""

    client_id: str
    local_dataset: Dataset
    dataset_size: int
    compute_capability: float = 1.0  # Relative compute power
    privacy_preference: str = "balanced"  # strict, balanced, relaxed

    def evaluate_locally(self, task: Task, adapter: Adapter, metrics: List[Metric]) -> ClientUpdate:
        """Perform local evaluation on client's dataset."""
        # Run evaluation on local dataset
        from ..core import evaluate

        # Create a subset of the local dataset for efficiency
        local_examples = list(self.local_dataset)[: min(100, len(self.local_dataset))]

        # Evaluate locally
        results = evaluate(task, adapter, local_examples, metrics)

        # Extract metrics
        local_metrics = {}
        for metric in metrics:
            if hasattr(results, "metrics"):
                local_metrics.update(results.metrics)
            else:
                # Fallback: compute metrics directly
                predictions = [getattr(ex, "prediction", "") for ex in local_examples]
                references = [getattr(ex, "reference", "") for ex in local_examples]
                local_metrics.update(metric.compute(predictions, references))

        return ClientUpdate(
            client_id=self.client_id,
            local_metrics=local_metrics,
            local_dataset_size=self.dataset_size,
            timestamp=time.time(),
        )


class FederatedDataset(Dataset):
    """Dataset that supports federated partitioning."""

    def __init__(
        self, base_dataset: Dataset, num_clients: int = 5, partitioning: str = "iid"
    ):  # iid, non-iid, dirichlet
        self.base_dataset = base_dataset
        self.num_clients = num_clients
        self.partitioning = partitioning
        self.client_datasets: Dict[str, List[Any]] = {}

        self._partition_dataset()

    def _partition_dataset(self):
        """Partition the dataset among clients."""
        all_examples = list(self.base_dataset)

        if self.partitioning == "iid":
            # Independent and identically distributed
            import random

            random.shuffle(all_examples)
            examples_per_client = len(all_examples) // self.num_clients

            for i in range(self.num_clients):
                start_idx = i * examples_per_client
                end_idx = (
                    (i + 1) * examples_per_client if i < self.num_clients - 1 else len(all_examples)
                )
                self.client_datasets[f"client_{i}"] = all_examples[start_idx:end_idx]

        elif self.partitioning == "non-iid":
            # Non-IID partitioning (simplified)
            # In practice, would use more sophisticated non-IID partitioning
            import random

            random.shuffle(all_examples)
            examples_per_client = len(all_examples) // self.num_clients

            for i in range(self.num_clients):
                start_idx = i * examples_per_client
                end_idx = (
                    (i + 1) * examples_per_client if i < self.num_clients - 1 else len(all_examples)
                )
                self.client_datasets[f"client_{i}"] = all_examples[start_idx:end_idx]

    def get_client_dataset(self, client_id: str) -> List[Any]:
        """Get dataset for a specific client."""
        return self.client_datasets.get(client_id, [])

    def __iter__(self):
        """Iterate over all examples (for compatibility)."""
        for examples in self.client_datasets.values():
            yield from examples

    def __len__(self):
        """Total number of examples."""
        return sum(len(examples) for examples in self.client_datasets.values())
