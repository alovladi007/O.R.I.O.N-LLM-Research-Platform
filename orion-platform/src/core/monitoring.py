"""
ORION Performance Monitoring and Bottleneck Analysis
===================================================

Real-time system monitoring, performance tracking, and bottleneck detection.
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
    """System performance metrics snapshot"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage: float = 0.0
    network_io_sent_mb: float = 0.0
    network_io_recv_mb: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    gpu_temperature: float = 0.0
    processing_time: float = 0.0
    throughput: float = 0.0
    queue_size: int = 0
    active_threads: int = 0
    error_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Bottleneck:
    """Detected system bottleneck"""
    component: str
    severity: str  # low, medium, high, critical
    metric: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    recommendation: str = ""
    
    def __str__(self) -> str:
        return (f"[{self.severity.upper()}] {self.component} bottleneck: "
                f"{self.metric}={self.current_value:.2f} (threshold={self.threshold:.2f})")


class PerformanceMonitor:
    """
    Real-time performance monitoring for ORION system.
    
    Features:
    - System resource monitoring (CPU, memory, disk, network)
    - GPU monitoring (if available)
    - Custom metric tracking
    - Performance history
    - Prometheus integration
    """
    
    def __init__(self, history_window: int = 1000, sampling_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            history_window: Number of historical metrics to keep
            sampling_interval: Seconds between metric samples
        """
        self.history_window = history_window
        self.sampling_interval = sampling_interval
        self.metrics_history: deque = deque(maxlen=history_window)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        
        # Performance counters
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Network I/O baseline
        self._network_baseline = psutil.net_io_counters()
        
        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.prom_cpu_usage = Gauge('orion_cpu_usage_percent', 'CPU usage percentage')
        self.prom_memory_usage = Gauge('orion_memory_usage_percent', 'Memory usage percentage')
        self.prom_gpu_usage = Gauge('orion_gpu_usage_percent', 'GPU usage percentage')
        self.prom_request_duration = Histogram('orion_request_duration_seconds', 
                                               'Request duration in seconds',
                                               buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0))
        self.prom_error_counter = Counter('orion_errors_total', 'Total number of errors')
        self.prom_active_threads = Gauge('orion_active_threads', 'Number of active threads')
    
    def start(self):
        """Start monitoring thread"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop monitoring thread"""
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join()
            logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Update Prometheus metrics if available
                if PROMETHEUS_AVAILABLE:
                    self._update_prometheus_metrics(metrics)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        metrics = PerformanceMetrics()
        
        # CPU metrics
        metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.memory_usage = memory.percent
        metrics.memory_available_gb = memory.available / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.disk_usage = disk.percent
        
        # Network I/O
        net_io = psutil.net_io_counters()
        metrics.network_io_sent_mb = (net_io.bytes_sent - self._network_baseline.bytes_sent) / (1024**2)
        metrics.network_io_recv_mb = (net_io.bytes_recv - self._network_baseline.bytes_recv) / (1024**2)
        
        # GPU metrics (if available)
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    metrics.gpu_usage = gpu.load * 100
                    metrics.gpu_memory = gpu.memoryUtil * 100
                    metrics.gpu_temperature = gpu.temperature
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
        
        # Thread count
        metrics.active_threads = threading.active_count()
        
        # Calculate latency percentiles if we have timer data
        if 'request' in self.timers and self.timers['request']:
            latencies = sorted(self.timers['request'][-100:])  # Last 100 requests
            metrics.latency_p50 = np.percentile(latencies, 50)
            metrics.latency_p95 = np.percentile(latencies, 95)
            metrics.latency_p99 = np.percentile(latencies, 99)
        
        # Calculate error rate
        total_requests = self.counters.get('requests', 0)
        if total_requests > 0:
            metrics.error_rate = (self.counters.get('errors', 0) / total_requests) * 100
        
        return metrics
    
    def _update_prometheus_metrics(self, metrics: PerformanceMetrics):
        """Update Prometheus metrics"""
        self.prom_cpu_usage.set(metrics.cpu_usage)
        self.prom_memory_usage.set(metrics.memory_usage)
        self.prom_gpu_usage.set(metrics.gpu_usage)
        self.prom_active_threads.set(metrics.active_threads)
    
    def track_custom_metric(self, name: str, value: float):
        """Track a custom metric"""
        self.custom_metrics[name].append((datetime.now(), value))
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric"""
        self.counters[name] += value
    
    def record_timing(self, name: str, duration: float):
        """Record a timing measurement"""
        self.timers[name].append(duration)
        
        # Keep only recent timings
        if len(self.timers[name]) > 1000:
            self.timers[name] = self.timers[name][-1000:]
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get summary statistics for recent metrics"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Calculate summary statistics
        summary = {
            'window_minutes': window_minutes,
            'sample_count': len(recent_metrics),
            'cpu_usage': {
                'mean': np.mean([m.cpu_usage for m in recent_metrics]),
                'max': max(m.cpu_usage for m in recent_metrics),
                'min': min(m.cpu_usage for m in recent_metrics),
            },
            'memory_usage': {
                'mean': np.mean([m.memory_usage for m in recent_metrics]),
                'max': max(m.memory_usage for m in recent_metrics),
                'min': min(m.memory_usage for m in recent_metrics),
            },
            'gpu_usage': {
                'mean': np.mean([m.gpu_usage for m in recent_metrics]),
                'max': max(m.gpu_usage for m in recent_metrics),
                'min': min(m.gpu_usage for m in recent_metrics),
            },
            'error_rate': {
                'mean': np.mean([m.error_rate for m in recent_metrics]),
                'max': max(m.error_rate for m in recent_metrics),
            },
            'latency_ms': {
                'p50': np.mean([m.latency_p50 * 1000 for m in recent_metrics if m.latency_p50 > 0]),
                'p95': np.mean([m.latency_p95 * 1000 for m in recent_metrics if m.latency_p95 > 0]),
                'p99': np.mean([m.latency_p99 * 1000 for m in recent_metrics if m.latency_p99 > 0]),
            }
        }
        
        return summary
    
    def monitor_function(self, func: Callable) -> Callable:
        """Decorator to monitor function performance"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                self.increment_counter('requests')
                return result
                
            except Exception as e:
                self.increment_counter('errors')
                raise
                
            finally:
                duration = time.time() - start_time
                self.record_timing('request', duration)
                
                if PROMETHEUS_AVAILABLE:
                    self.prom_request_duration.observe(duration)
        
        return wrapper


class BottleneckAnalyzer:
    """
    Analyzes system performance to detect and diagnose bottlenecks.
    
    Features:
    - Real-time bottleneck detection
    - Trend analysis
    - Predictive warnings
    - Optimization recommendations
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize bottleneck analyzer.
        
        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor
        self.bottlenecks: List[Bottleneck] = []
        
        # Bottleneck thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'gpu_usage': {'warning': 85.0, 'critical': 95.0},
            'gpu_temperature': {'warning': 80.0, 'critical': 90.0},
            'disk_usage': {'warning': 80.0, 'critical': 90.0},
            'error_rate': {'warning': 5.0, 'critical': 10.0},
            'latency_p99': {'warning': 1.0, 'critical': 5.0},  # seconds
            'queue_size': {'warning': 100, 'critical': 500},
        }
        
        # Analysis state
        self._analyzing = False
        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_interval = 10.0  # seconds
    
    def start(self):
        """Start bottleneck analysis"""
        if not self._analyzing:
            self._analyzing = True
            self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self._analysis_thread.start()
            logger.info("Bottleneck analysis started")
    
    def stop(self):
        """Stop bottleneck analysis"""
        if self._analyzing:
            self._analyzing = False
            if self._analysis_thread:
                self._analysis_thread.join()
            logger.info("Bottleneck analysis stopped")
    
    def _analysis_loop(self):
        """Main analysis loop"""
        while self._analyzing:
            try:
                self.analyze()
                time.sleep(self._analysis_interval)
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
    
    def analyze(self) -> List[Bottleneck]:
        """Perform bottleneck analysis"""
        current_metrics = self.monitor.get_current_metrics()
        if not current_metrics:
            return []
        
        detected_bottlenecks = []
        
        # Check each metric against thresholds
        metric_checks = [
            ('cpu_usage', current_metrics.cpu_usage, 'CPU'),
            ('memory_usage', current_metrics.memory_usage, 'Memory'),
            ('gpu_usage', current_metrics.gpu_usage, 'GPU'),
            ('gpu_temperature', current_metrics.gpu_temperature, 'GPU Temperature'),
            ('disk_usage', current_metrics.disk_usage, 'Disk'),
            ('error_rate', current_metrics.error_rate, 'Error Rate'),
            ('latency_p99', current_metrics.latency_p99, 'Latency'),
        ]
        
        for metric_name, value, component in metric_checks:
            if metric_name in self.thresholds and value > 0:
                bottleneck = self._check_threshold(metric_name, value, component)
                if bottleneck:
                    detected_bottlenecks.append(bottleneck)
        
        # Check for trend-based bottlenecks
        trend_bottlenecks = self._analyze_trends()
        detected_bottlenecks.extend(trend_bottlenecks)
        
        # Update bottleneck list
        self.bottlenecks = detected_bottlenecks
        
        # Log critical bottlenecks
        for bottleneck in detected_bottlenecks:
            if bottleneck.severity == 'critical':
                logger.warning(str(bottleneck))
        
        return detected_bottlenecks
    
    def _check_threshold(self, metric: str, value: float, component: str) -> Optional[Bottleneck]:
        """Check if a metric exceeds thresholds"""
        thresholds = self.thresholds.get(metric, {})
        
        if value >= thresholds.get('critical', float('inf')):
            severity = 'critical'
            threshold = thresholds['critical']
        elif value >= thresholds.get('warning', float('inf')):
            severity = 'warning'
            threshold = thresholds['warning']
        else:
            return None
        
        # Generate recommendation
        recommendation = self._get_recommendation(metric, severity, value)
        
        return Bottleneck(
            component=component,
            severity=severity,
            metric=metric,
            current_value=value,
            threshold=threshold,
            recommendation=recommendation
        )
    
    def _analyze_trends(self) -> List[Bottleneck]:
        """Analyze metric trends for predictive bottlenecks"""
        bottlenecks = []
        
        # Get recent metrics
        summary = self.monitor.get_metrics_summary(window_minutes=5)
        if not summary:
            return bottlenecks
        
        # Check for sustained high usage
        if summary['cpu_usage']['mean'] > 60 and summary['cpu_usage']['min'] > 50:
            bottlenecks.append(Bottleneck(
                component='CPU',
                severity='warning',
                metric='cpu_usage_trend',
                current_value=summary['cpu_usage']['mean'],
                threshold=60.0,
                recommendation='Sustained high CPU usage detected. Consider scaling horizontally.'
            ))
        
        # Check for memory leak patterns
        recent_metrics = list(self.monitor.metrics_history)[-20:]
        if len(recent_metrics) >= 10:
            memory_values = [m.memory_usage for m in recent_metrics]
            if all(memory_values[i] <= memory_values[i+1] for i in range(len(memory_values)-1)):
                bottlenecks.append(Bottleneck(
                    component='Memory',
                    severity='warning',
                    metric='memory_leak',
                    current_value=memory_values[-1],
                    threshold=0,
                    recommendation='Possible memory leak detected. Monitor memory usage closely.'
                ))
        
        return bottlenecks
    
    def _get_recommendation(self, metric: str, severity: str, value: float) -> str:
        """Get optimization recommendation for a bottleneck"""
        recommendations = {
            'cpu_usage': {
                'warning': 'Consider optimizing CPU-intensive operations or adding more CPU cores.',
                'critical': 'Critical CPU usage! Scale horizontally or optimize algorithms immediately.'
            },
            'memory_usage': {
                'warning': 'Memory usage is high. Review memory allocations and consider caching strategies.',
                'critical': 'Critical memory usage! Risk of OOM. Increase memory or optimize usage.'
            },
            'gpu_usage': {
                'warning': 'GPU utilization is high. Consider batch size optimization or multi-GPU setup.',
                'critical': 'GPU at capacity! Reduce workload or add additional GPUs.'
            },
            'gpu_temperature': {
                'warning': 'GPU temperature is high. Check cooling system.',
                'critical': 'Critical GPU temperature! Thermal throttling likely. Improve cooling immediately.'
            },
            'disk_usage': {
                'warning': 'Disk space running low. Clean up old data or expand storage.',
                'critical': 'Critical disk usage! System may become unstable. Free up space immediately.'
            },
            'error_rate': {
                'warning': f'Error rate is {value:.1f}%. Investigate error logs.',
                'critical': f'Critical error rate ({value:.1f}%)! System reliability compromised.'
            },
            'latency_p99': {
                'warning': f'High latency detected ({value*1000:.0f}ms). Review slow operations.',
                'critical': f'Critical latency ({value*1000:.0f}ms)! User experience severely impacted.'
            }
        }
        
        return recommendations.get(metric, {}).get(severity, 'Monitor situation closely.')
    
    def get_bottleneck_report(self) -> Dict[str, Any]:
        """Generate comprehensive bottleneck report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'bottlenecks': [
                {
                    'component': b.component,
                    'severity': b.severity,
                    'metric': b.metric,
                    'current_value': b.current_value,
                    'threshold': b.threshold,
                    'recommendation': b.recommendation,
                    'timestamp': b.timestamp.isoformat()
                }
                for b in self.bottlenecks
            ],
            'summary': {
                'total': len(self.bottlenecks),
                'critical': sum(1 for b in self.bottlenecks if b.severity == 'critical'),
                'warning': sum(1 for b in self.bottlenecks if b.severity == 'warning'),
            },
            'system_metrics': self.monitor.get_metrics_summary()
        }
    
    def export_report(self, filepath: str):
        """Export bottleneck report to file"""
        report = self.get_bottleneck_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Bottleneck report exported to {filepath}")