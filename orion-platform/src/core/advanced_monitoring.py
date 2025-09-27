"""
ORION Advanced Performance Monitoring and Bottleneck Analysis
===========================================================

Complete implementation with all advanced features from the original ORION system.
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import wraps
import json
import numpy as np
import warnings

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Track system performance across different stages"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    processing_time: float = 0.0
    throughput: float = 0.0
    queue_size: int = 0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AdvancedBottleneckAnalyzer:
    """Advanced real-time system bottleneck detection and analysis"""
    
    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        self.metrics_history = defaultdict(lambda: deque(maxlen=history_window))
        self.bottleneck_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'gpu_usage': 90.0,
            'gpu_memory': 85.0,
            'error_rate': 5.0,
            'queue_size': 50
        }
        
        # Analysis state
        self._analyzing = False
        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_interval = 10.0  # seconds
        
    def monitor_performance(func):
        """Decorator for monitoring function performance"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            start_cpu = psutil.cpu_percent()
            start_memory = psutil.virtual_memory().percent
            
            # GPU metrics if available
            gpu_usage = gpu_memory = 0.0
            try:
                if GPU_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage = gpus[0].load * 100
                        gpu_memory = gpus[0].memoryUtil * 100
            except:
                pass
            
            try:
                result = func(self, *args, **kwargs)
                error_occurred = False
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                error_occurred = True
                raise
            finally:
                end_time = time.time()
                processing_time = end_time - start_time
                
                metrics = PerformanceMetrics(
                    cpu_usage=psutil.cpu_percent() - start_cpu,
                    memory_usage=psutil.virtual_memory().percent - start_memory,
                    gpu_usage=gpu_usage,
                    gpu_memory=gpu_memory,
                    processing_time=processing_time,
                    error_rate=1.0 if error_occurred else 0.0
                )
                
                self.record_metrics(func.__name__, metrics)
                
            return result
        return wrapper
    
    def record_metrics(self, stage: str, metrics: PerformanceMetrics):
        """Record performance metrics for a stage"""
        for key, value in metrics.__dict__.items():
            if key != 'timestamp':
                self.metrics_history[f"{stage}_{key}"].append(value)
    
    def detect_bottlenecks(self) -> Dict[str, List[str]]:
        """Detect current system bottlenecks"""
        bottlenecks = defaultdict(list)
        
        for metric_key, history in self.metrics_history.items():
            if not history:
                continue
                
            stage, metric = metric_key.rsplit('_', 1)
            if metric in self.bottleneck_thresholds:
                recent_avg = np.mean(list(history)[-10:])  # Last 10 measurements
                threshold = self.bottleneck_thresholds[metric]
                
                if recent_avg > threshold:
                    bottlenecks[stage].append(f"{metric}: {recent_avg:.1f}% (threshold: {threshold}%)")
        
        return dict(bottlenecks)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'bottlenecks': self.detect_bottlenecks(),
            'stage_performance': {},
            'recommendations': []
        }
        
        # Aggregate by stage
        stages = set(key.rsplit('_', 1)[0] for key in self.metrics_history.keys())
        for stage in stages:
            stage_metrics = {}
            for metric in ['cpu_usage', 'memory_usage', 'processing_time', 'error_rate']:
                key = f"{stage}_{metric}"
                if key in self.metrics_history and self.metrics_history[key]:
                    values = list(self.metrics_history[key])
                    stage_metrics[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'p95': np.percentile(values, 95),
                        'recent_trend': np.mean(values[-5:]) - np.mean(values[-15:-5]) if len(values) >= 15 else 0
                    }
            report['stage_performance'][stage] = stage_metrics
        
        # Generate recommendations
        bottlenecks = report['bottlenecks']
        if bottlenecks:
            for stage, issues in bottlenecks.items():
                if 'cpu_usage' in str(issues):
                    report['recommendations'].append(f"Scale {stage} horizontally or optimize CPU-bound operations")
                if 'memory_usage' in str(issues):
                    report['recommendations'].append(f"Increase memory for {stage} or implement memory optimization")
                if 'gpu' in str(issues):
                    report['recommendations'].append(f"Add GPU resources for {stage} or optimize batch sizes")
        
        return report
    
    def start(self):
        """Start bottleneck analysis"""
        if not self._analyzing:
            self._analyzing = True
            self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self._analysis_thread.start()
            logger.info("Advanced bottleneck analysis started")
    
    def stop(self):
        """Stop bottleneck analysis"""
        if self._analyzing:
            self._analyzing = False
            if self._analysis_thread:
                self._analysis_thread.join()
            logger.info("Advanced bottleneck analysis stopped")
    
    def _analysis_loop(self):
        """Main analysis loop"""
        while self._analyzing:
            try:
                report = self.get_performance_report()
                
                # Log critical bottlenecks
                if report['bottlenecks']:
                    logger.warning(f"Bottlenecks detected: {report['bottlenecks']}")
                
                # Log recommendations
                for rec in report['recommendations']:
                    logger.info(f"Performance recommendation: {rec}")
                
                time.sleep(self._analysis_interval)
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        utilization = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'open_files': len(psutil.Process().open_files()),
        }
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    utilization['gpu_percent'] = gpus[0].load * 100
                    utilization['gpu_memory_percent'] = gpus[0].memoryUtil * 100
                    utilization['gpu_temperature'] = gpus[0].temperature
            except:
                pass
        
        return utilization
    
    def predict_resource_exhaustion(self, resource: str = 'memory') -> Optional[timedelta]:
        """Predict when a resource might be exhausted based on trends"""
        key_mapping = {
            'memory': 'memory_usage',
            'cpu': 'cpu_usage',
            'gpu': 'gpu_usage'
        }
        
        if resource not in key_mapping:
            return None
        
        # Look for the metric across all stages
        all_values = []
        timestamps = []
        current_time = time.time()
        
        for metric_key, history in self.metrics_history.items():
            if key_mapping[resource] in metric_key and history:
                all_values.extend(list(history))
                # Estimate timestamps based on position in history
                for i, _ in enumerate(history):
                    timestamps.append(current_time - (len(history) - i) * 10)  # Assume 10s intervals
        
        if len(all_values) < 10:
            return None
        
        # Simple linear regression for trend
        x = np.array(timestamps[-20:])  # Last 20 points
        y = np.array(all_values[-20:])
        
        if len(x) < 2:
            return None
        
        # Fit line
        coeffs = np.polyfit(x - x[0], y, 1)
        slope = coeffs[0]
        
        if slope <= 0:  # Resource usage is decreasing
            return None
        
        # Predict when it will hit 100%
        current_value = y[-1]
        time_to_100 = (100 - current_value) / (slope * 60)  # Convert to minutes
        
        if time_to_100 > 0 and time_to_100 < 1440:  # Within 24 hours
            return timedelta(minutes=int(time_to_100))
        
        return None