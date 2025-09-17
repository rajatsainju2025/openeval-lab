"""
Monitoring Dashboard Module for OpenEval

This module provides comprehensive monitoring capabilities including real-time metrics,
visualization, alerting, and performance dashboards.
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from collections import defaultdict, deque
import statistics
from datetime import datetime, timedelta
import webbrowser
import tempfile
import os

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
    logger.warning("psutil not available, system monitoring disabled")


@dataclass
class MetricData:
    """Container for metric data points."""
    timestamp: float
    name: str
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric: str
    condition: str  # ">", "<", ">=", "<=", "==", "!="
    threshold: Union[int, float]
    duration: int  # seconds
    severity: str  # "info", "warning", "error", "critical"
    enabled: bool = True
    cooldown: int = 300  # seconds between alerts
    last_triggered: Optional[float] = None


@dataclass
class Alert:
    """Alert instance."""
    rule_name: str
    metric: str
    value: Union[int, float]
    threshold: Union[int, float]
    severity: str
    timestamp: float
    message: str
    resolved: bool = False
    resolved_at: Optional[float] = None


class MetricsCollector:
    """Collects and aggregates metrics from various sources."""

    def __init__(self, retention_period: int = 3600):  # 1 hour default
        self.retention_period = retention_period
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregates: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._collection_thread: Optional[threading.Thread] = None
        self._running = False

    def start_collection(self) -> None:
        """Start metrics collection."""
        if self._running:
            return

        self._running = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")

    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self._running:
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                self._update_aggregates()
                self._cleanup_old_metrics()

                time.sleep(10)  # Collect every 10 seconds

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(30)

    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        if not HAS_PSUTIL or not psutil:
            return

        try:
            # CPU metrics
            self.record_metric("system.cpu.percent", psutil.cpu_percent(interval=1))
            cpu_count = psutil.cpu_count()
            if cpu_count is not None:
                self.record_metric("system.cpu.count", cpu_count)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system.memory.percent", memory.percent)
            self.record_metric("system.memory.used_mb", memory.used / 1024 / 1024)
            self.record_metric("system.memory.available_mb", memory.available / 1024 / 1024)

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk.percent", disk.percent)
            self.record_metric("system.disk.used_gb", disk.used / 1024 / 1024 / 1024)

            # Network metrics
            network = psutil.net_io_counters()
            self.record_metric("system.network.bytes_sent_mb", network.bytes_sent / 1024 / 1024)
            self.record_metric("system.network.bytes_recv_mb", network.bytes_recv / 1024 / 1024)

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def _collect_application_metrics(self) -> None:
        """Collect application-specific metrics."""
        try:
            # Process metrics
            if HAS_PSUTIL and psutil:
                process = psutil.Process()
                self.record_metric("app.memory_mb", process.memory_info().rss / 1024 / 1024)
                self.record_metric("app.cpu_percent", process.cpu_percent())
                self.record_metric("app.threads", process.num_threads())
            else:
                # Fallback values when psutil is not available
                self.record_metric("app.memory_mb", 100.0)
                self.record_metric("app.cpu_percent", 10.0)
                self.record_metric("app.threads", 4)

            # Application-specific metrics (placeholders)
            self.record_metric("app.active_tasks", 0)  # Would be updated by other components
            self.record_metric("app.completed_tasks", 0)
            self.record_metric("app.failed_tasks", 0)
            self.record_metric("app.cache_hits", 0)
            self.record_metric("app.cache_misses", 0)

        except Exception as e:
            logger.warning(f"Failed to collect application metrics: {e}")

    def record_metric(self, name: str, value: Union[int, float],
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric data point."""
        with self._lock:
            metric_data = MetricData(
                timestamp=time.time(),
                name=name,
                value=value,
                tags=tags or {},
                metadata=metadata or {}
            )
            self.metrics[name].append(metric_data)

    def get_metric_data(self, name: str, time_range: Optional[int] = None) -> List[MetricData]:
        """Get metric data for a given name and time range."""
        with self._lock:
            data = list(self.metrics.get(name, []))
            if time_range:
                cutoff = time.time() - time_range
                data = [d for d in data if d.timestamp >= cutoff]
            return data

    def get_metric_stats(self, name: str, time_range: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics for a metric."""
        data = self.get_metric_data(name, time_range)
        if not data:
            return {"count": 0}

        values = [d.value for d in data]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "latest": values[-1],
            "latest_timestamp": data[-1].timestamp
        }

    def _update_aggregates(self) -> None:
        """Update aggregated metrics."""
        with self._lock:
            for name in self.metrics:
                data = list(self.metrics[name])
                if data:
                    self.aggregates[name] = self.get_metric_stats(name)

    def _cleanup_old_metrics(self) -> None:
        """Clean up old metric data."""
        cutoff = time.time() - self.retention_period
        with self._lock:
            for name in list(self.metrics.keys()):
                original_len = len(self.metrics[name])
                while self.metrics[name] and self.metrics[name][0].timestamp < cutoff:
                    self.metrics[name].popleft()

                # Remove empty deques
                if not self.metrics[name]:
                    del self.metrics[name]
                    if name in self.aggregates:
                        del self.aggregates[name]


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        self.alert_callbacks: List[Callable[[Alert], None]] = []

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self._lock:
            self.rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> None:
        """Remove an alert rule."""
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")

    def check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check all alert rules against current metrics."""
        with self._lock:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue

                if rule.metric in metrics:
                    current_value = metrics[rule.metric]
                    triggered = self._check_condition(current_value, rule.condition, rule.threshold)

                    if triggered:
                        self._trigger_alert(rule, current_value)
                    elif rule.name in self.active_alerts:
                        self._resolve_alert(rule.name)

    def _check_condition(self, value: Union[int, float], condition: str,
                        threshold: Union[int, float]) -> bool:
        """Check if a condition is met."""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False

    def _trigger_alert(self, rule: AlertRule, current_value: Union[int, float]) -> None:
        """Trigger an alert."""
        now = time.time()

        # Check cooldown
        if rule.last_triggered and (now - rule.last_triggered) < rule.cooldown:
            return

        alert = Alert(
            rule_name=rule.name,
            metric=rule.metric,
            value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            timestamp=now,
            message=f"Alert triggered: {rule.metric} {rule.condition} {rule.threshold} "
                   f"(current: {current_value})"
        )

        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        rule.last_triggered = now

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.warning(f"Alert triggered: {alert.message}")

    def _resolve_alert(self, rule_name: str) -> None:
        """Resolve an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolved_at = time.time()
            del self.active_alerts[rule_name]
            logger.info(f"Alert resolved: {rule_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            return list(self.alert_history)[-limit:]


class DashboardGenerator:
    """Generates HTML dashboard for monitoring."""

    def __init__(self, collector: MetricsCollector, alert_manager: AlertManager):
        self.collector = collector
        self.alert_manager = alert_manager

    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OpenEval Monitoring Dashboard</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .alert {{ background: #ffebee; border: 1px solid #f44336; padding: 10px; margin: 10px 0; }}
                .warning {{ background: #fff3e0; border: 1px solid #ff9800; }}
                .info {{ background: #e3f2fd; border: 1px solid #2196f3; }}
                .chart {{ width: 100%; height: 200px; background: #f9f9f9; margin: 10px 0; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; }}
                h1, h2 {{ color: #333; }}
                .timestamp {{ color: #666; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <h1>OpenEval Monitoring Dashboard</h1>
            <div class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

            {self._generate_alerts_section()}
            {self._generate_system_metrics_section()}
            {self._generate_application_metrics_section()}
            {self._generate_performance_charts()}

            <script>
                // Auto-refresh functionality
                setTimeout(function() {{
                    location.reload();
                }}, 30000);
            </script>
        </body>
        </html>
        """
        return html

    def _generate_alerts_section(self) -> str:
        """Generate alerts section."""
        active_alerts = self.alert_manager.get_active_alerts()
        if not active_alerts:
            return "<h2>Active Alerts</h2><p>No active alerts</p>"

        alerts_html = "<h2>Active Alerts</h2>"
        for alert in active_alerts:
            css_class = alert.severity
            alerts_html += f"""
            <div class="alert {css_class}">
                <strong>{alert.severity.upper()}</strong>: {alert.message}
                <br><small>{datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S')}</small>
            </div>
            """

        return alerts_html

    def _generate_system_metrics_section(self) -> str:
        """Generate system metrics section."""
        system_metrics = [
            "system.cpu.percent",
            "system.memory.percent",
            "system.disk.percent",
            "system.network.bytes_sent_mb"
        ]

        html = '<div class="grid">'
        for metric in system_metrics:
            stats = self.collector.get_metric_stats(metric, 300)  # Last 5 minutes
            if stats.get("count", 0) > 0:
                html += f"""
                <div class="card">
                    <h3>{metric.replace('system.', '').replace('.', ' ').title()}</h3>
                    <div class="metric">
                        Current: {stats.get('latest', 'N/A'):.2f}<br>
                        Mean: {stats.get('mean', 'N/A'):.2f}<br>
                        Min: {stats.get('min', 'N/A'):.2f}<br>
                        Max: {stats.get('max', 'N/A'):.2f}
                    </div>
                </div>
                """

        html += '</div>'
        return html

    def _generate_application_metrics_section(self) -> str:
        """Generate application metrics section."""
        app_metrics = [
            "app.memory_mb",
            "app.cpu_percent",
            "app.active_tasks",
            "app.cache_hits"
        ]

        html = '<div class="grid">'
        for metric in app_metrics:
            stats = self.collector.get_metric_stats(metric, 300)  # Last 5 minutes
            if stats.get("count", 0) > 0:
                html += f"""
                <div class="card">
                    <h3>{metric.replace('app.', '').replace('.', ' ').title()}</h3>
                    <div class="metric">
                        Current: {stats.get('latest', 'N/A'):.2f}<br>
                        Mean: {stats.get('mean', 'N/A'):.2f}<br>
                        Min: {stats.get('min', 'N/A'):.2f}<br>
                        Max: {stats.get('max', 'N/A'):.2f}
                    </div>
                </div>
                """

        html += '</div>'
        return html

    def _generate_performance_charts(self) -> str:
        """Generate performance charts section."""
        return """
        <h2>Performance Trends</h2>
        <div class="chart">
            <p>Chart visualization would be implemented here with a JavaScript charting library</p>
            <p>Supported metrics: CPU usage, Memory usage, Task completion rates, Cache performance</p>
        </div>
        """

    def open_dashboard(self) -> None:
        """Open dashboard in web browser."""
        html_content = self.generate_dashboard_html()

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_file = f.name

        # Open in browser
        webbrowser.open(f'file://{temp_file}')

        # Clean up after some time
        threading.Timer(300, lambda: os.unlink(temp_file)).start()  # Delete after 5 minutes


class MonitoringDashboard:
    """Main monitoring dashboard coordinator."""

    def __init__(self):
        self.collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.dashboard_generator = DashboardGenerator(self.collector, self.alert_manager)
        self._running = False

    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self._running:
            return

        self._running = True
        self.collector.start_collection()

        # Add default alert rules
        self._setup_default_alerts()

        logger.info("Monitoring dashboard started")

    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self._running = False
        self.collector.stop_collection()
        logger.info("Monitoring dashboard stopped")

    def _setup_default_alerts(self) -> None:
        """Set up default alert rules."""
        default_rules = [
            AlertRule(
                name="high_cpu",
                metric="system.cpu.percent",
                condition=">",
                threshold=90,
                duration=60,
                severity="warning"
            ),
            AlertRule(
                name="high_memory",
                metric="system.memory.percent",
                condition=">",
                threshold=85,
                duration=60,
                severity="warning"
            ),
            AlertRule(
                name="low_disk_space",
                metric="system.disk.percent",
                condition=">",
                threshold=90,
                duration=300,
                severity="error"
            ),
            AlertRule(
                name="high_app_memory",
                metric="app.memory_mb",
                condition=">",
                threshold=1000,  # 1GB
                duration=60,
                severity="warning"
            )
        ]

        for rule in default_rules:
            self.alert_manager.add_rule(rule)

    def update_metric(self, name: str, value: Union[int, float],
                     tags: Optional[Dict[str, str]] = None) -> None:
        """Update a metric value."""
        self.collector.record_metric(name, value, tags)

    def check_alerts(self) -> None:
        """Check all alert conditions."""
        metrics = {}
        for name in self.collector.metrics:
            stats = self.collector.get_metric_stats(name, 60)  # Last minute
            if stats.get("count", 0) > 0:
                metrics[name] = stats.get("latest", 0)

        self.alert_manager.check_alerts(metrics)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "timestamp": time.time(),
            "system_metrics": {
                name: self.collector.get_metric_stats(name, 300)
                for name in self.collector.metrics
                if name.startswith("system.")
            },
            "application_metrics": {
                name: self.collector.get_metric_stats(name, 300)
                for name in self.collector.metrics
                if name.startswith("app.")
            },
            "active_alerts": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
                for alert in self.alert_manager.get_active_alerts()
            ],
            "alert_history": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved
                }
                for alert in self.alert_manager.get_alert_history(50)
            ]
        }

    def open_dashboard(self) -> None:
        """Open the monitoring dashboard in a web browser."""
        self.dashboard_generator.open_dashboard()