"""Advanced result aggregation and analysis tools."""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class RunSummary:
    """Summary of a single evaluation run."""
    run_id: str
    task: str
    adapter: str
    dataset: str
    metrics: Dict[str, Any]
    size: int
    timestamp: Optional[str] = None
    file_path: Optional[str] = None
    run_name: Optional[str] = None


class ResultAggregator:
    """Aggregates and analyzes multiple evaluation runs."""
    
    def __init__(self):
        self.runs: List[RunSummary] = []
    
    def load_run(self, file_path: Path) -> Optional[RunSummary]:
        """Load a single run from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract key information
            run_id = file_path.stem
            task = data.get('task', 'unknown')
            adapter = data.get('adapter', 'unknown') 
            dataset = data.get('dataset', 'unknown')
            metrics = data.get('metrics', {})
            size = data.get('size', 0)
            timestamp = data.get('timestamp')
            run_name = data.get('run_name')
            
            return RunSummary(
                run_id=run_id,
                task=task,
                adapter=adapter,
                dataset=dataset,
                metrics=metrics,
                size=size,
                timestamp=timestamp,
                file_path=str(file_path),
                run_name=run_name
            )
            
        except Exception as e:
            print(f"Failed to load run from {file_path}: {e}")
            return None
    
    def load_runs_from_directory(self, directory: Path, pattern: str = "*.json") -> int:
        """Load all runs from a directory."""
        loaded_count = 0
        
        for file_path in directory.glob(pattern):
            if file_path.name.startswith('index') or file_path.name.startswith('summary'):
                continue  # Skip aggregation files
                
            run_summary = self.load_run(file_path)
            if run_summary:
                self.runs.append(run_summary)
                loaded_count += 1
        
        return loaded_count
    
    def filter_runs(
        self,
        task: Optional[str] = None,
        adapter: Optional[str] = None,
        dataset: Optional[str] = None,
        min_size: Optional[int] = None
    ) -> List[RunSummary]:
        """Filter runs by criteria with optimized single-pass filtering."""
        # Use generator expression for memory-efficient single-pass filtering
        return [
            r for r in self.runs
            if (not task or task in r.task) and
               (not adapter or adapter in r.adapter) and
               (not dataset or dataset in r.dataset) and
               (min_size is None or r.size >= min_size)
        ]
    
    def get_metric_values(self, runs: List[RunSummary], metric_name: str) -> List[float]:
        """Extract metric values from runs."""
        values = []
        
        for run in runs:
            # Navigate nested metric structure
            metric_data = run.metrics.get(metric_name, {})
            
            if isinstance(metric_data, dict):
                # Look for common metric field names
                for key in ['accuracy', 'f1', 'score', 'value', metric_name]:
                    if key in metric_data:
                        try:
                            values.append(float(metric_data[key]))
                            break
                        except (ValueError, TypeError):
                            continue
            elif isinstance(metric_data, (int, float)):
                values.append(float(metric_data))
        
        return values
    
    def generate_leaderboard(
        self,
        metric_name: str,
        task_filter: Optional[str] = None,
        dataset_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate leaderboard for a specific metric."""
        
        runs = self.filter_runs(task=task_filter, dataset=dataset_filter)
        
        leaderboard = []
        for run in runs:
            metric_values = self.get_metric_values([run], metric_name)
            if metric_values:
                leaderboard.append({
                    'rank': 0,  # Will be set after sorting
                    'run_id': run.run_id,
                    'adapter': run.adapter,
                    'task': run.task,
                    'dataset': run.dataset,
                    'metric_value': metric_values[0],
                    'size': run.size,
                    'run_name': run.run_name or run.run_id,
                    'timestamp': run.timestamp
                })
        
        # Sort by metric value (descending)
        leaderboard.sort(key=lambda x: x['metric_value'], reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(leaderboard):
            entry['rank'] = i + 1
        
        if top_k:
            leaderboard = leaderboard[:top_k]
        
        return leaderboard
    
    def compare_adapters(
        self,
        metric_name: str,
        task_filter: Optional[str] = None,
        dataset_filter: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare adapter performance across runs."""
        
        runs = self.filter_runs(task=task_filter, dataset=dataset_filter)
        
        adapter_stats = {}
        
        for run in runs:
            adapter = run.adapter
            metric_values = self.get_metric_values([run], metric_name)
            
            if metric_values and adapter not in adapter_stats:
                adapter_stats[adapter] = {
                    'values': [],
                    'run_count': 0,
                    'total_examples': 0
                }
            
            if metric_values:
                adapter_stats[adapter]['values'].extend(metric_values)
                adapter_stats[adapter]['run_count'] += 1
                adapter_stats[adapter]['total_examples'] += run.size
        
        # Compute statistics
        for adapter, stats in adapter_stats.items():
            values = stats['values']
            if values:
                stats['mean'] = statistics.mean(values)
                stats['median'] = statistics.median(values)
                stats['std'] = statistics.stdev(values) if len(values) > 1 else 0.0
                stats['min'] = min(values)
                stats['max'] = max(values)
                stats['best_value'] = max(values)  # Assuming higher is better
        
        return adapter_stats
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        
        if not self.runs:
            return {"error": "No runs loaded"}
        
        # Basic statistics
        total_runs = len(self.runs)
        unique_tasks = len(set(run.task for run in self.runs))
        unique_adapters = len(set(run.adapter for run in self.runs))
        unique_datasets = len(set(run.dataset for run in self.runs))
        total_examples = sum(run.size for run in self.runs)
        
        # Metric coverage
        all_metrics = set()
        for run in self.runs:
            all_metrics.update(run.metrics.keys())
        
        # Recent activity
        recent_runs = sorted(
            [r for r in self.runs if r.timestamp],
            key=lambda x: x.timestamp or "",
            reverse=True
        )[:5]
        
        return {
            "summary": {
                "total_runs": total_runs,
                "unique_tasks": unique_tasks,
                "unique_adapters": unique_adapters,
                "unique_datasets": unique_datasets,
                "total_examples": total_examples,
                "available_metrics": sorted(list(all_metrics))
            },
            "recent_runs": [
                {
                    "run_id": r.run_id,
                    "adapter": r.adapter,
                    "task": r.task,
                    "timestamp": r.timestamp
                }
                for r in recent_runs
            ],
            "adapters": sorted(list(set(run.adapter for run in self.runs))),
            "tasks": sorted(list(set(run.task for run in self.runs))),
            "datasets": sorted(list(set(run.dataset for run in self.runs)))
        }
    
    def export_to_csv(
        self,
        output_path: Path,
        metric_name: Optional[str] = None,
        include_all_metrics: bool = False
    ):
        """Export runs to CSV format."""
        
        if not self.runs:
            raise ValueError("No runs to export")
        
        rows = []
        
        for run in self.runs:
            row = {
                'run_id': run.run_id,
                'task': run.task,
                'adapter': run.adapter,
                'dataset': run.dataset,
                'size': run.size,
                'timestamp': run.timestamp,
                'run_name': run.run_name,
                'file_path': run.file_path
            }
            
            if metric_name:
                values = self.get_metric_values([run], metric_name)
                row[metric_name] = values[0] if values else None
            
            if include_all_metrics:
                for metric, data in run.metrics.items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            row[f"{metric}_{key}"] = value
                    else:
                        row[metric] = data
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        return len(rows)
    
    def find_best_runs(
        self,
        metric_name: str,
        group_by: str = "task",  # task, adapter, dataset
        top_k: int = 1
    ) -> Dict[str, List[RunSummary]]:
        """Find best performing runs grouped by specified criterion."""
        
        groups = {}
        
        for run in self.runs:
            group_key = getattr(run, group_by, "unknown")
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append(run)
        
        best_runs = {}
        
        for group_key, group_runs in groups.items():
            # Get metric values for all runs in group
            runs_with_values = []
            
            for run in group_runs:
                values = self.get_metric_values([run], metric_name)
                if values:
                    runs_with_values.append((run, values[0]))
            
            # Sort by metric value and take top k
            runs_with_values.sort(key=lambda x: x[1], reverse=True)
            best_runs[group_key] = [run for run, _ in runs_with_values[:top_k]]
        
        return best_runs
    
    def detect_regressions(
        self,
        metric_name: str,
        baseline_run_id: str,
        threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Detect performance regressions compared to baseline."""
        
        baseline_run = None
        for run in self.runs:
            if run.run_id == baseline_run_id:
                baseline_run = run
                break
        
        if not baseline_run:
            raise ValueError(f"Baseline run {baseline_run_id} not found")
        
        baseline_values = self.get_metric_values([baseline_run], metric_name)
        if not baseline_values:
            raise ValueError(f"Baseline run has no {metric_name} metric")
        
        baseline_value = baseline_values[0]
        regressions = []
        
        for run in self.runs:
            if run.run_id == baseline_run_id:
                continue
            
            current_values = self.get_metric_values([run], metric_name)
            if not current_values:
                continue
            
            current_value = current_values[0]
            relative_change = (current_value - baseline_value) / baseline_value
            
            if relative_change < -threshold:  # Negative change beyond threshold
                regressions.append({
                    'run_id': run.run_id,
                    'adapter': run.adapter,
                    'baseline_value': baseline_value,
                    'current_value': current_value,
                    'relative_change': relative_change,
                    'absolute_change': current_value - baseline_value
                })
        
        return sorted(regressions, key=lambda x: x['relative_change'])


def create_analysis_dashboard(aggregator: ResultAggregator, output_dir: Path):
    """Create a simple analysis dashboard."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate summary report
    summary = aggregator.generate_summary_report()
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Export all runs to CSV
    aggregator.export_to_csv(
        output_dir / "all_runs.csv",
        include_all_metrics=True
    )
    
    # Generate leaderboards for common metrics
    common_metrics = ['accuracy', 'f1', 'rouge_l', 'bleu']
    
    for metric in common_metrics:
        try:
            leaderboard = aggregator.generate_leaderboard(metric, top_k=10)
            if leaderboard:
                with open(output_dir / f"leaderboard_{metric}.json", 'w') as f:
                    json.dump(leaderboard, f, indent=2)
        except Exception:
            continue
    
    # Adapter comparison
    for metric in common_metrics:
        try:
            adapter_stats = aggregator.compare_adapters(metric)
            if adapter_stats:
                with open(output_dir / f"adapter_comparison_{metric}.json", 'w') as f:
                    json.dump(adapter_stats, f, indent=2)
        except Exception:
            continue
    
    print(f"Analysis dashboard created in {output_dir}")
    print(f"- Summary: {output_dir / 'summary.json'}")
    print(f"- All runs: {output_dir / 'all_runs.csv'}")
    print(f"- Leaderboards and comparisons: {output_dir / 'leaderboard_*.json'}")
