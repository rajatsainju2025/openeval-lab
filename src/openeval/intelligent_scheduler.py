"""
Intelligent Evaluation Scheduler with ML-driven adaptation.

This module implements an adaptive evaluation scheduler that learns from historical
performance patterns to optimize evaluation workflows. It uses machine learning
to predict optimal batch sizes, concurrency levels, and resource allocation.
"""

import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .core import Task, Dataset, Adapter, Metric
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationProfile:
    """Profile for a specific evaluation configuration."""

    task_type: str
    dataset_size: int
    adapter_type: str
    metric_count: int
    avg_latency_ms: float
    throughput_eps: float
    memory_usage_mb: float
    success_rate: float
    optimal_batch_size: int
    optimal_concurrency: int
    timestamp: datetime

    def to_features(self) -> np.ndarray:
        """Convert profile to feature vector for ML."""
        return np.array(
            [
                self.dataset_size,
                self.metric_count,
                self.avg_latency_ms,
                self.memory_usage_mb,
                hash(self.task_type) % 1000,  # Hash to numeric
                hash(self.adapter_type) % 1000,
            ]
        )


@dataclass
class SchedulingDecision:
    """ML-driven scheduling decision."""

    recommended_batch_size: int
    recommended_concurrency: int
    estimated_duration_minutes: float
    confidence_score: float
    resource_prediction: Dict[str, float]


class PerformancePredictor:
    """ML model for predicting evaluation performance."""

    def __init__(self):
        self.throughput_model = None
        self.latency_model = None
        self.memory_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False

    def train(self, profiles: List[EvaluationProfile]) -> bool:
        """Train the performance prediction models."""
        if not SKLEARN_AVAILABLE or len(profiles) < 10:
            logger.warning("Insufficient data or sklearn unavailable for ML training")
            return False

        try:
            # Prepare training data
            X = np.array([profile.to_features() for profile in profiles])
            y_throughput = np.array([profile.throughput_eps for profile in profiles])
            y_latency = np.array([profile.avg_latency_ms for profile in profiles])
            y_memory = np.array([profile.memory_usage_mb for profile in profiles])

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train models
            self.throughput_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.latency_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.memory_model = RandomForestRegressor(n_estimators=50, random_state=42)

            self.throughput_model.fit(X_scaled, y_throughput)
            self.latency_model.fit(X_scaled, y_latency)
            self.memory_model.fit(X_scaled, y_memory)

            self.is_trained = True
            logger.info(f"Trained performance models on {len(profiles)} profiles")
            return True

        except Exception as e:
            logger.error(f"Failed to train performance models: {e}")
            return False

    def predict(
        self, task_type: str, dataset_size: int, adapter_type: str, metric_count: int
    ) -> Optional[Dict[str, float]]:
        """Predict performance metrics for given configuration."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return None

        try:
            # Create feature vector
            features = np.array(
                [
                    [
                        dataset_size,
                        metric_count,
                        0,  # avg_latency_ms (unknown)
                        0,  # memory_usage_mb (unknown)
                        hash(task_type) % 1000,
                        hash(adapter_type) % 1000,
                    ]
                ]
            )

            features_scaled = self.scaler.transform(features)

            # Make predictions
            throughput_pred = self.throughput_model.predict(features_scaled)[0]
            latency_pred = self.latency_model.predict(features_scaled)[0]
            memory_pred = self.memory_model.predict(features_scaled)[0]

            return {
                "throughput_eps": max(0.1, throughput_pred),
                "avg_latency_ms": max(10, latency_pred),
                "memory_usage_mb": max(50, memory_pred),
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None


class EvaluationHistory:
    """Persistent storage for evaluation performance history."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".openeval" / "scheduler.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    dataset_size INTEGER NOT NULL,
                    adapter_type TEXT NOT NULL,
                    metric_count INTEGER NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    throughput_eps REAL NOT NULL,
                    memory_usage_mb REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    optimal_batch_size INTEGER NOT NULL,
                    optimal_concurrency INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_adapter ON evaluation_profiles(task_type, adapter_type)"
            )

    def save_profile(self, profile: EvaluationProfile):
        """Save evaluation profile to history."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO evaluation_profiles
                (task_type, dataset_size, adapter_type, metric_count, avg_latency_ms,
                 throughput_eps, memory_usage_mb, success_rate, optimal_batch_size,
                 optimal_concurrency, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    profile.task_type,
                    profile.dataset_size,
                    profile.adapter_type,
                    profile.metric_count,
                    profile.avg_latency_ms,
                    profile.throughput_eps,
                    profile.memory_usage_mb,
                    profile.success_rate,
                    profile.optimal_batch_size,
                    profile.optimal_concurrency,
                    profile.timestamp.isoformat(),
                ),
            )

    def get_recent_profiles(self, days: int = 30) -> List[EvaluationProfile]:
        """Get recent evaluation profiles."""
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM evaluation_profiles
                WHERE datetime(timestamp) > ?
                ORDER BY timestamp DESC
            """,
                (cutoff.isoformat(),),
            )

            profiles = []
            for row in cursor.fetchall():
                profiles.append(
                    EvaluationProfile(
                        task_type=row[1],
                        dataset_size=row[2],
                        adapter_type=row[3],
                        metric_count=row[4],
                        avg_latency_ms=row[5],
                        throughput_eps=row[6],
                        memory_usage_mb=row[7],
                        success_rate=row[8],
                        optimal_batch_size=row[9],
                        optimal_concurrency=row[10],
                        timestamp=datetime.fromisoformat(row[11]),
                    )
                )

            return profiles

    def get_similar_profiles(
        self, task_type: str, adapter_type: str, dataset_size: int, limit: int = 10
    ) -> List[EvaluationProfile]:
        """Get profiles similar to the given configuration."""
        size_tolerance = 0.2  # 20% tolerance on dataset size

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM evaluation_profiles
                WHERE task_type = ? AND adapter_type = ?
                AND dataset_size BETWEEN ? AND ?
                ORDER BY ABS(dataset_size - ?) ASC
                LIMIT ?
            """,
                (
                    task_type,
                    adapter_type,
                    int(dataset_size * (1 - size_tolerance)),
                    int(dataset_size * (1 + size_tolerance)),
                    dataset_size,
                    limit,
                ),
            )

            profiles = []
            for row in cursor.fetchall():
                profiles.append(
                    EvaluationProfile(
                        task_type=row[1],
                        dataset_size=row[2],
                        adapter_type=row[3],
                        metric_count=row[4],
                        avg_latency_ms=row[5],
                        throughput_eps=row[6],
                        memory_usage_mb=row[7],
                        success_rate=row[8],
                        optimal_batch_size=row[9],
                        optimal_concurrency=row[10],
                        timestamp=datetime.fromisoformat(row[11]),
                    )
                )

            return profiles


class IntelligentScheduler:
    """Intelligent evaluation scheduler with ML-driven optimization."""

    def __init__(self, history_db: Optional[Path] = None):
        self.history = EvaluationHistory(history_db)
        self.predictor = PerformancePredictor()
        self.active_evaluations: Dict[str, Dict] = {}
        self.lock = threading.Lock()

        # Load and train models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize and train ML models from historical data."""
        recent_profiles = self.history.get_recent_profiles(days=90)

        if len(recent_profiles) >= 10:
            success = self.predictor.train(recent_profiles)
            if success:
                logger.info(
                    f"Initialized ML models with {len(recent_profiles)} historical profiles"
                )
            else:
                logger.warning("Failed to initialize ML models, using heuristics")
        else:
            logger.info(
                f"Insufficient historical data ({len(recent_profiles)} profiles), using heuristics"
            )

    def optimize_scheduling(
        self, task: Task, dataset: Dataset, adapter: Adapter, metrics: List[Metric]
    ) -> SchedulingDecision:
        """Generate optimal scheduling parameters using ML and historical data."""

        # Extract configuration details
        task_type = task.__class__.__name__
        adapter_type = adapter.__class__.__name__
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else 1000
        metric_count = len(metrics)

        logger.info(
            f"Optimizing schedule for {task_type} + {adapter_type} on {dataset_size} examples"
        )

        # Try ML prediction first
        ml_prediction = self.predictor.predict(task_type, dataset_size, adapter_type, metric_count)

        # Get similar historical profiles
        similar_profiles = self.history.get_similar_profiles(
            task_type, adapter_type, dataset_size, limit=5
        )

        # Combine ML predictions with historical heuristics
        if ml_prediction and similar_profiles:
            decision = self._combine_ml_and_heuristics(
                ml_prediction, similar_profiles, dataset_size
            )
        elif similar_profiles:
            decision = self._heuristic_scheduling(similar_profiles, dataset_size)
        else:
            decision = self._default_scheduling(dataset_size)

        logger.info(
            f"Scheduling decision: batch_size={decision.recommended_batch_size}, "
            f"concurrency={decision.recommended_concurrency}, "
            f"estimated_duration={decision.estimated_duration_minutes:.1f}min"
        )

        return decision

    def _combine_ml_and_heuristics(
        self, ml_pred: Dict[str, float], profiles: List[EvaluationProfile], dataset_size: int
    ) -> SchedulingDecision:
        """Combine ML predictions with historical heuristics."""

        # ML-based estimates
        predicted_throughput = ml_pred["throughput_eps"]
        predicted_latency = ml_pred["avg_latency_ms"]
        predicted_memory = ml_pred["memory_usage_mb"]

        # Historical averages
        if profiles:
            hist_batch_sizes = [p.optimal_batch_size for p in profiles]
            hist_concurrency = [p.optimal_concurrency for p in profiles]
            avg_batch_size = sum(hist_batch_sizes) / len(hist_batch_sizes)
            avg_concurrency = sum(hist_concurrency) / len(hist_concurrency)
        else:
            avg_batch_size = 32
            avg_concurrency = 4

        # Optimize batch size based on memory constraints
        # Assume 4GB available memory as baseline
        available_memory_mb = 4000
        memory_per_batch = (
            predicted_memory / avg_batch_size if avg_batch_size > 0 else predicted_memory / 32
        )
        max_batch_by_memory = max(1, int(available_memory_mb * 0.7 / memory_per_batch))

        # Optimize batch size for throughput
        optimal_batch_size = min(
            max_batch_by_memory,
            max(1, int(avg_batch_size * 1.2)),  # 20% above historical average
            min(256, dataset_size // 4),  # Don't exceed 1/4 of dataset or 256
        )

        # Optimize concurrency based on latency
        if predicted_latency > 1000:  # High latency
            optimal_concurrency = min(8, max(2, int(avg_concurrency * 1.5)))
        else:  # Low latency
            optimal_concurrency = min(4, max(1, int(avg_concurrency)))

        # Estimate duration
        effective_throughput = predicted_throughput * optimal_concurrency * 0.8  # 80% efficiency
        estimated_duration_minutes = (dataset_size / effective_throughput) / 60

        # Confidence score based on data quality
        confidence = min(1.0, len(profiles) / 10.0)  # Max confidence with 10+ profiles

        return SchedulingDecision(
            recommended_batch_size=optimal_batch_size,
            recommended_concurrency=optimal_concurrency,
            estimated_duration_minutes=estimated_duration_minutes,
            confidence_score=confidence,
            resource_prediction={
                "memory_mb": predicted_memory,
                "throughput_eps": effective_throughput,
                "latency_ms": predicted_latency,
            },
        )

    def _heuristic_scheduling(
        self, profiles: List[EvaluationProfile], dataset_size: int
    ) -> SchedulingDecision:
        """Use historical heuristics when ML is not available."""

        # Calculate averages from similar profiles
        avg_throughput = sum(p.throughput_eps for p in profiles) / len(profiles)
        avg_latency = sum(p.avg_latency_ms for p in profiles) / len(profiles)
        avg_memory = sum(p.memory_usage_mb for p in profiles) / len(profiles)
        avg_batch_size = sum(p.optimal_batch_size for p in profiles) / len(profiles)
        avg_concurrency = sum(p.optimal_concurrency for p in profiles) / len(profiles)

        # Adjust based on dataset size ratio
        size_ratios = [dataset_size / p.dataset_size for p in profiles]
        avg_size_ratio = sum(size_ratios) / len(size_ratios)

        recommended_batch_size = max(1, int(avg_batch_size * (avg_size_ratio**0.3)))
        recommended_concurrency = max(1, int(avg_concurrency))

        estimated_duration_minutes = (
            dataset_size / (avg_throughput * recommended_concurrency)
        ) / 60

        return SchedulingDecision(
            recommended_batch_size=recommended_batch_size,
            recommended_concurrency=recommended_concurrency,
            estimated_duration_minutes=estimated_duration_minutes,
            confidence_score=0.7,  # Medium confidence with heuristics
            resource_prediction={
                "memory_mb": avg_memory * avg_size_ratio,
                "throughput_eps": avg_throughput * recommended_concurrency,
                "latency_ms": avg_latency,
            },
        )

    def _default_scheduling(self, dataset_size: int) -> SchedulingDecision:
        """Default scheduling when no historical data is available."""

        # Conservative defaults based on dataset size
        if dataset_size < 100:
            batch_size, concurrency = 16, 2
        elif dataset_size < 1000:
            batch_size, concurrency = 32, 4
        elif dataset_size < 10000:
            batch_size, concurrency = 64, 6
        else:
            batch_size, concurrency = 128, 8

        # Conservative estimates
        estimated_throughput = 2.0  # 2 examples per second
        estimated_duration_minutes = (dataset_size / (estimated_throughput * concurrency)) / 60

        return SchedulingDecision(
            recommended_batch_size=batch_size,
            recommended_concurrency=concurrency,
            estimated_duration_minutes=estimated_duration_minutes,
            confidence_score=0.3,  # Low confidence with defaults
            resource_prediction={
                "memory_mb": 512.0,
                "throughput_eps": estimated_throughput * concurrency,
                "latency_ms": 500.0,
            },
        )

    def record_evaluation_performance(
        self,
        task: Task,
        dataset: Dataset,
        adapter: Adapter,
        metrics: List[Metric],
        performance_stats: Dict[str, Any],
        used_batch_size: int,
        used_concurrency: int,
    ):
        """Record the actual performance of an evaluation for future learning."""

        timing = performance_stats.get("timing", {})
        stats = performance_stats.get("stats", {})

        profile = EvaluationProfile(
            task_type=task.__class__.__name__,
            dataset_size=(
                len(dataset) if hasattr(dataset, "__len__") else performance_stats.get("size", 1000)
            ),
            adapter_type=adapter.__class__.__name__,
            metric_count=len(metrics),
            avg_latency_ms=timing.get("avg_latency_ms", 0.0),
            throughput_eps=timing.get("throughput_eps", 0.0),
            memory_usage_mb=stats.get("peak_memory_mb", 0.0),
            success_rate=1.0 - stats.get("error_rate", 0.0),
            optimal_batch_size=used_batch_size,
            optimal_concurrency=used_concurrency,
            timestamp=datetime.now(),
        )

        try:
            self.history.save_profile(profile)
            logger.info(
                f"Recorded performance profile for {profile.task_type} + {profile.adapter_type}"
            )

            # Retrain models periodically
            if len(self.history.get_recent_profiles(days=7)) % 25 == 0:  # Every 25 evaluations
                self._retrain_models()

        except Exception as e:
            logger.error(f"Failed to record performance profile: {e}")

    def _retrain_models(self):
        """Retrain ML models with updated data."""
        try:
            recent_profiles = self.history.get_recent_profiles(days=90)
            if len(recent_profiles) >= 10:
                success = self.predictor.train(recent_profiles)
                if success:
                    logger.info(f"Retrained ML models with {len(recent_profiles)} profiles")
        except Exception as e:
            logger.error(f"Failed to retrain models: {e}")

    def get_performance_insights(self) -> Dict[str, Any]:
        """Get insights about evaluation performance patterns."""
        profiles = self.history.get_recent_profiles(days=30)

        if not profiles:
            return {"message": "No recent evaluation data available"}

        # Group by task and adapter types
        task_stats = {}
        adapter_stats = {}

        for profile in profiles:
            # Task statistics
            if profile.task_type not in task_stats:
                task_stats[profile.task_type] = {
                    "count": 0,
                    "total_throughput": 0,
                    "total_latency": 0,
                    "total_memory": 0,
                    "success_rates": [],
                }

            stats = task_stats[profile.task_type]
            stats["count"] += 1
            stats["total_throughput"] += profile.throughput_eps
            stats["total_latency"] += profile.avg_latency_ms
            stats["total_memory"] += profile.memory_usage_mb
            stats["success_rates"].append(profile.success_rate)

            # Adapter statistics
            if profile.adapter_type not in adapter_stats:
                adapter_stats[profile.adapter_type] = {
                    "count": 0,
                    "total_throughput": 0,
                    "total_latency": 0,
                    "success_rates": [],
                }

            astats = adapter_stats[profile.adapter_type]
            astats["count"] += 1
            astats["total_throughput"] += profile.throughput_eps
            astats["total_latency"] += profile.avg_latency_ms
            astats["success_rates"].append(profile.success_rate)

        # Calculate averages
        for task_type, stats in task_stats.items():
            count = stats["count"]
            stats["avg_throughput"] = stats["total_throughput"] / count
            stats["avg_latency"] = stats["total_latency"] / count
            stats["avg_memory"] = stats["total_memory"] / count
            stats["avg_success_rate"] = sum(stats["success_rates"]) / count
            del stats["total_throughput"], stats["total_latency"], stats["total_memory"]

        for adapter_type, stats in adapter_stats.items():
            count = stats["count"]
            stats["avg_throughput"] = stats["total_throughput"] / count
            stats["avg_latency"] = stats["total_latency"] / count
            stats["avg_success_rate"] = sum(stats["success_rates"]) / count
            del stats["total_throughput"], stats["total_latency"]

        return {
            "total_evaluations": len(profiles),
            "task_performance": task_stats,
            "adapter_performance": adapter_stats,
            "ml_model_trained": self.predictor.is_trained,
            "data_quality_score": min(1.0, len(profiles) / 50.0),  # Score out of 1.0
        }


# Factory function for easy integration
def create_intelligent_scheduler(history_db: Optional[Path] = None) -> IntelligentScheduler:
    """Create an intelligent scheduler instance."""
    return IntelligentScheduler(history_db)


# CLI command for scheduler insights
def scheduler_insights_command():
    """CLI command to show scheduler performance insights."""
    scheduler = create_intelligent_scheduler()
    insights = scheduler.get_performance_insights()

    print("📊 Intelligent Scheduler Performance Insights")
    print("=" * 50)
    print(f"Total Evaluations: {insights.get('total_evaluations', 0)}")
    print(f"ML Model Trained: {'✅' if insights.get('ml_model_trained') else '❌'}")
    print(f"Data Quality Score: {insights.get('data_quality_score', 0):.2f}/1.0")
    print()

    if "task_performance" in insights:
        print("🎯 Task Performance:")
        for task, stats in insights["task_performance"].items():
            print(f"  {task}:")
            print(f"    - Evaluations: {stats['count']}")
            print(f"    - Avg Throughput: {stats['avg_throughput']:.2f} eps")
            print(f"    - Avg Latency: {stats['avg_latency']:.1f} ms")
            print(f"    - Success Rate: {stats['avg_success_rate']:.1%}")

    if "adapter_performance" in insights:
        print("\n🤖 Adapter Performance:")
        for adapter, stats in insights["adapter_performance"].items():
            print(f"  {adapter}:")
            print(f"    - Evaluations: {stats['count']}")
            print(f"    - Avg Throughput: {stats['avg_throughput']:.2f} eps")
            print(f"    - Avg Latency: {stats['avg_latency']:.1f} ms")
            print(f"    - Success Rate: {stats['avg_success_rate']:.1%}")
