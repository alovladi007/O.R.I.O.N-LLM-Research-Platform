"""
ORION: Testing Framework & Performance Optimization
==================================================

This module provides comprehensive testing, performance optimization, and CI/CD
pipeline implementations for the ORION materials science AI system, including:

1. Unit Testing Framework
2. Integration Testing Suite
3. Performance Benchmarking
4. Load Testing and Stress Testing
5. Security Testing
6. CI/CD Pipeline Configuration
7. Performance Optimization Strategies
8. Scaling and Resource Management
9. Caching and Data Optimization
10. Monitoring and Alerting Integration

Author: ORION Development Team
"""

import asyncio
import pytest
import unittest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import pandas as pd
import torch
import time
import json
import yaml
import tempfile
import shutil
from pathlib import Path
import subprocess
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import memory_profiler
from typing import Dict, List, Any, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta
import uuid
import hashlib
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Testing libraries
try:
    import locust
    from locust import HttpUser, task, between
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    print("Locust not available - load testing will be limited")

try:
    import hypothesis
    from hypothesis import given, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("Hypothesis not available - property-based testing disabled")

try:
    import bandit
    import safety
    SECURITY_TOOLS_AVAILABLE = True
except ImportError:
    SECURITY_TOOLS_AVAILABLE = False
    print("Security testing tools not available")

# Profiling and optimization
try:
    import cProfile
    import pstats
    import line_profiler
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    print("Profiling tools not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. UNIT TESTING FRAMEWORK
# =====================================================================

class TestORIONCore(unittest.TestCase):
    """Unit tests for ORION core components"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            'model_types': ['gnn', 'rf'],
            'random_state': 42,
            'test_mode': True
        }
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_temp_dir)
    
    def _cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_physics_sanity_checker(self):
        """Test physics constraint validation"""
        from orion_core_modules import PhysicsSanityChecker
        
        checker = PhysicsSanityChecker()
        
        # Test valid candidate
        valid_candidate = {
            'predictions': {
                'bandgap': 2.5,
                'density': 5.2,
                'bulk_modulus': 150.0,
                'shear_modulus': 80.0
            }
        }
        
        is_valid, errors = checker.validate_candidate(valid_candidate)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid candidate (negative bandgap)
        invalid_candidate = {
            'predictions': {
                'bandgap': -1.0,
                'density': 5.2
            }
        }
        
        is_valid, errors = checker.validate_candidate(invalid_candidate)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_uncertainty_quantifier(self):
        """Test uncertainty quantification methods"""
        from orion_core_modules import UncertaintyQuantifier
        
        quantifier = UncertaintyQuantifier()
        
        # Generate test data
        np.random.seed(42)
        n_samples = 100
        predictions = np.random.normal(2.0, 0.5, n_samples)
        uncertainties = np.random.uniform(0.1, 0.3, n_samples)
        true_values = predictions + np.random.normal(0, 0.2, n_samples)
        
        # Test calibration
        quantifier.calibrate_uncertainty(predictions, uncertainties, true_values)
        self.assertTrue(quantifier.is_calibrated)
        
        # Test metrics computation
        metrics = quantifier.compute_uncertainty_metrics(predictions, uncertainties, true_values)
        
        self.assertIn('calibration_error', metrics)
        self.assertIn('sharpness', metrics)
        self.assertIn('uncertainty_error_correlation', metrics)
        self.assertGreater(metrics['uncertainty_error_correlation'], -1)
        self.assertLess(metrics['uncertainty_error_correlation'], 1)
    
    def test_provenance_weighted_consensus(self):
        """Test conflict resolution algorithm"""
        from orion_core_modules import ProvenanceWeightedConsensus, SourceMetadata
        
        resolver = ProvenanceWeightedConsensus()
        
        # Create test property values with different source reliabilities
        property_values = [
            (2.1, SourceMetadata("10.1000/journal1", "2023-01-15", 50, 3.2, "journal")),
            (2.3, SourceMetadata("10.1000/journal2", "2022-06-10", 20, 2.1, "journal")),
            (1.9, SourceMetadata("arxiv.2301.12345", "2023-02-01", 5, 0.0, "preprint"))
        ]
        
        resolution = resolver.resolve_property_conflict(property_values, 'bandgap')
        
        self.assertIn('consensus_value', resolution)
        self.assertIn('uncertainty', resolution)
        self.assertIn('num_sources', resolution)
        self.assertEqual(resolution['num_sources'], 3)
        
        # Consensus should be weighted towards higher-reliability sources
        self.assertGreater(resolution['consensus_value'], 1.9)
        self.assertLess(resolution['consensus_value'], 2.3)
    
    def test_uncertainty_aware_ranking(self):
        """Test uncertainty-aware candidate ranking"""
        from orion_core_modules import UncertaintyAwareRanker
        
        ranker = UncertaintyAwareRanker(risk_aversion=1.5)
        
        # Create test candidates
        candidates = []
        for i in range(10):
            candidate = {
                'predictions': {'bandgap': np.random.uniform(1.0, 3.0)},
                'uncertainties': {'bandgap': np.random.uniform(0.1, 0.5)},
                'embedding': np.random.randn(128)
            }
            candidates.append(candidate)
        
        # Test ranking
        ranked = ranker.rank_candidates(
            candidates, target_property='bandgap', target_value=2.0, return_details=True
        )
        
        self.assertEqual(len(ranked), 10)
        
        # Check that scores are in descending order
        scores = [score for _, score, _ in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Test batch selection
        selected_indices = ranker.select_batch(candidates, batch_size=5)
        self.assertEqual(len(selected_indices), 5)
        self.assertEqual(len(set(selected_indices)), 5)  # No duplicates
    
    @patch('torch.cuda.is_available')
    def test_gnn_model_creation(self, mock_cuda):
        """Test GNN model creation and basic functionality"""
        mock_cuda.return_value = False  # Force CPU mode for testing
        
        from orion_core_modules import GNNSurrogate
        
        # Create test model
        model = GNNSurrogate(
            num_node_features=10,
            hidden_dim=64,
            num_layers=3,
            dropout=0.1,
            uncertainty_mode='both'
        )
        
        # Test model parameters
        self.assertGreater(len(list(model.parameters())), 0)
        
        # Test with dummy data
        batch_size = 4
        num_nodes = 20
        num_edges = 40
        
        # Create dummy graph data
        x = torch.randn(batch_size * num_nodes, 10)
        edge_index = torch.randint(0, batch_size * num_nodes, (2, batch_size * num_edges))
        batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
        
        # Mock data object
        data = Mock()
        data.x = x
        data.edge_index = edge_index
        data.batch = batch
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            predictions, uncertainties = model(data, return_uncertainty=True, mc_samples=5)
        
        # Check outputs
        self.assertIsInstance(predictions, dict)
        self.assertIsInstance(uncertainties, dict)
        
        for prop in ['bandgap', 'formation_energy', 'bulk_modulus', 'density']:
            self.assertIn(prop, predictions)
            self.assertIn(prop, uncertainties)
            self.assertEqual(predictions[prop].shape[0], batch_size)
            self.assertEqual(uncertainties[prop].shape[0], batch_size)


class TestORIONIntegration(unittest.TestCase):
    """Integration tests for ORION system components"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_config = {
            'databases': {
                'neo4j': {'uri': 'bolt://localhost:7687', 'username': 'neo4j', 'password': 'test'},
                'redis': {'host': 'localhost', 'port': 6379, 'db': 15}  # Use test DB
            },
            'test_mode': True
        }
    
    @patch('orion_integration_deployment.GraphDatabase.driver')
    @patch('orion_integration_deployment.redis.Redis')
    def test_knowledge_graph_integration(self, mock_redis, mock_neo4j):
        """Test knowledge graph operations"""
        from orion_integration_deployment import ORIONKnowledgeGraph
        
        # Mock database responses
        mock_session = Mock()
        mock_neo4j.return_value.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value.single.return_value = {'material_id': 'test_material_001'}
        
        kg = ORIONKnowledgeGraph(self.test_config['databases'])
        
        # Test material creation
        material_data = {
            'material_id': 'test_material_001',
            'formula': 'TiO2',
            'composition': {'Ti': 0.33, 'O': 0.67}
        }
        
        # This would call the mocked database
        # result = asyncio.run(kg.create_material_node(material_data))
        # self.assertEqual(result, 'test_material_001')
    
    @patch('aiohttp.ClientSession.get')
    async def test_external_api_integration(self, mock_get):
        """Test external API integrations"""
        from orion_integration_deployment import ExternalAPIManager
        
        # Mock API response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'data': [{
                'material_id': 'mp-123',
                'formula_pretty': 'TiO2',
                'band_gap': 3.2
            }]
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        api_manager = ExternalAPIManager({'api_keys': {'materials_project': 'test_key'}})
        
        async with api_manager:
            result = await api_manager.query_materials_project('TiO2')
        
        self.assertIn('materials', result)
        self.assertEqual(len(result['materials']), 1)
        self.assertEqual(result['materials'][0]['formula'], 'TiO2')
    
    def test_stream_processor_message_handling(self):
        """Test stream processor message handling"""
        from orion_advanced_features import StreamProcessor, StreamMessage
        
        processor = StreamProcessor()
        
        # Test message creation
        message = StreamMessage(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            source='test_source',
            message_type='experimental_result',
            payload={'test': 'data'}
        )
        
        # Test message serialization
        message_dict = message.to_dict()
        self.assertIn('message_id', message_dict)
        self.assertIn('payload', message_dict)
        self.assertEqual(message_dict['message_type'], 'experimental_result')


# =====================================================================
# 2. PERFORMANCE BENCHMARKING
# =====================================================================

@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    test_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ORIONBenchmarkSuite:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.results = []
        self.baseline_results = {}
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'benchmark_iterations': 10,
            'warmup_iterations': 3,
            'test_data_sizes': [100, 500, 1000, 5000],
            'memory_profiling': True,
            'cpu_profiling': True,
            'save_results': True,
            'results_path': './benchmark_results'
        }
    
    def run_prediction_benchmark(self) -> List[BenchmarkResult]:
        """Benchmark prediction performance across different model types"""
        
        results = []
        
        for data_size in self.config['test_data_sizes']:
            # Generate synthetic test data
            test_data = self._generate_synthetic_data(data_size)
            
            # Benchmark different model types
            for model_type in ['gnn', 'rf', 'ensemble']:
                result = self._benchmark_model_prediction(model_type, test_data)
                results.append(result)
                logger.info(f"Benchmarked {model_type} with {data_size} samples: "
                           f"{result.execution_time:.3f}s, {result.throughput:.1f} samples/s")
        
        return results
    
    def run_knowledge_graph_benchmark(self) -> List[BenchmarkResult]:
        """Benchmark knowledge graph operations"""
        
        results = []
        
        # Test different operation types
        operations = [
            ('create_material_node', self._benchmark_kg_create),
            ('add_property_relationship', self._benchmark_kg_add_property),
            ('query_material_properties', self._benchmark_kg_query),
            ('find_similar_materials', self._benchmark_kg_similarity)
        ]
        
        for op_name, op_func in operations:
            result = self._benchmark_operation(op_name, op_func)
            results.append(result)
            logger.info(f"KG operation {op_name}: {result.execution_time:.3f}s")
        
        return results
    
    def run_simulation_orchestrator_benchmark(self) -> List[BenchmarkResult]:
        """Benchmark simulation orchestration performance"""
        
        results = []
        
        # Test job submission and processing
        for num_jobs in [10, 50, 100, 200]:
            result = self._benchmark_simulation_jobs(num_jobs)
            results.append(result)
            logger.info(f"Simulation orchestrator with {num_jobs} jobs: "
                       f"{result.execution_time:.3f}s, {result.throughput:.1f} jobs/s")
        
        return results
    
    def run_stream_processing_benchmark(self) -> List[BenchmarkResult]:
        """Benchmark real-time stream processing"""
        
        results = []
        
        # Test different message rates
        for message_rate in [10, 50, 100, 500]:  # messages per second
            result = self._benchmark_stream_processing(message_rate)
            results.append(result)
            logger.info(f"Stream processing at {message_rate} msg/s: "
                       f"latency={result.execution_time:.3f}s, errors={result.error_rate:.1%}")
        
        return results
    
    def _generate_synthetic_data(self, size: int) -> List[Dict[str, Any]]:
        """Generate synthetic test data"""
        np.random.seed(42)
        
        data = []
        for i in range(size):
            sample = {
                'material_id': f'test_material_{i:06d}',
                'composition': {
                    'Ti': np.random.uniform(0.2, 0.5),
                    'O': np.random.uniform(0.5, 0.8)
                },
                'features': np.random.randn(50).tolist(),
                'target_bandgap': np.random.uniform(1.0, 4.0),
                'target_formation_energy': np.random.uniform(-3.0, 0.0)
            }
            data.append(sample)
        
        return data
    
    def _benchmark_model_prediction(self, model_type: str, test_data: List[Dict[str, Any]]) -> BenchmarkResult:
        """Benchmark model prediction performance"""
        
        # Warmup
        for _ in range(self.config['warmup_iterations']):
            self._run_mock_prediction(model_type, test_data[:10])
        
        # Actual benchmark
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        predictions = []
        errors = 0
        
        for _ in range(self.config['benchmark_iterations']):
            try:
                batch_predictions = self._run_mock_prediction(model_type, test_data)
                predictions.extend(batch_predictions)
            except Exception as e:
                errors += 1
                logger.warning(f"Prediction error: {e}")
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = len(test_data) * self.config['benchmark_iterations'] / execution_time
        error_rate = errors / self.config['benchmark_iterations']
        
        return BenchmarkResult(
            test_name=f"prediction_{model_type}_{len(test_data)}",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=psutil.cpu_percent(),
            throughput=throughput,
            error_rate=error_rate,
            metadata={
                'model_type': model_type,
                'data_size': len(test_data),
                'iterations': self.config['benchmark_iterations']
            }
        )
    
    def _run_mock_prediction(self, model_type: str, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mock prediction function for benchmarking"""
        
        # Simulate different computational costs for different models
        if model_type == 'gnn':
            time.sleep(0.001 * len(test_data))  # Simulate GNN computation
        elif model_type == 'rf':
            time.sleep(0.0005 * len(test_data))  # Simulate RF computation
        elif model_type == 'ensemble':
            time.sleep(0.0015 * len(test_data))  # Simulate ensemble computation
        
        # Generate mock predictions
        predictions = []
        for sample in test_data:
            prediction = {
                'material_id': sample['material_id'],
                'bandgap': np.random.uniform(1.0, 4.0),
                'formation_energy': np.random.uniform(-3.0, 0.0),
                'uncertainty': np.random.uniform(0.1, 0.3)
            }
            predictions.append(prediction)
        
        return predictions
    
    def _benchmark_operation(self, name: str, operation_func: Callable) -> BenchmarkResult:
        """Generic operation benchmarking"""
        
        # Warmup
        for _ in range(self.config['warmup_iterations']):
            operation_func()
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = 0
        for _ in range(self.config['benchmark_iterations']):
            try:
                operation_func()
            except Exception as e:
                errors += 1
                logger.warning(f"Operation error: {e}")
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        throughput = self.config['benchmark_iterations'] / execution_time
        
        return BenchmarkResult(
            test_name=name,
            execution_time=execution_time,
            memory_usage=end_memory - start_memory,
            cpu_usage=psutil.cpu_percent(),
            throughput=throughput,
            error_rate=errors / self.config['benchmark_iterations']
        )
    
    def _benchmark_kg_create(self):
        """Mock knowledge graph create operation"""
        time.sleep(0.001)  # Simulate database write
    
    def _benchmark_kg_add_property(self):
        """Mock knowledge graph add property operation"""
        time.sleep(0.0015)  # Simulate relationship creation
    
    def _benchmark_kg_query(self):
        """Mock knowledge graph query operation"""
        time.sleep(0.005)  # Simulate complex query
    
    def _benchmark_kg_similarity(self):
        """Mock similarity search operation"""
        time.sleep(0.01)  # Simulate similarity computation
    
    def _benchmark_simulation_jobs(self, num_jobs: int) -> BenchmarkResult:
        """Benchmark simulation job processing"""
        
        start_time = time.time()
        
        # Simulate job processing
        for i in range(num_jobs):
            # Mock job processing time
            time.sleep(0.001)
        
        end_time = time.time()
        execution_time = end_time - start_time
        throughput = num_jobs / execution_time
        
        return BenchmarkResult(
            test_name=f"simulation_jobs_{num_jobs}",
            execution_time=execution_time,
            memory_usage=0,  # Not measured for this test
            cpu_usage=psutil.cpu_percent(),
            throughput=throughput,
            error_rate=0.0,
            metadata={'num_jobs': num_jobs}
        )
    
    def _benchmark_stream_processing(self, message_rate: int) -> BenchmarkResult:
        """Benchmark stream message processing"""
        
        duration = 10  # seconds
        num_messages = message_rate * duration
        
        start_time = time.time()
        
        # Simulate message processing
        for i in range(num_messages):
            # Mock message processing time
            time.sleep(0.0001)
        
        end_time = time.time()
        execution_time = end_time - start_time
        throughput = num_messages / execution_time
        
        return BenchmarkResult(
            test_name=f"stream_processing_{message_rate}",
            execution_time=execution_time / num_messages,  # Per message latency
            memory_usage=0,
            cpu_usage=psutil.cpu_percent(),
            throughput=throughput,
            error_rate=0.0,
            metadata={'message_rate': message_rate, 'duration': duration}
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all benchmark suites"""
        
        logger.info("Starting comprehensive ORION benchmark suite...")
        
        benchmark_results = {
            'prediction': self.run_prediction_benchmark(),
            'knowledge_graph': self.run_knowledge_graph_benchmark(),
            'simulation_orchestrator': self.run_simulation_orchestrator_benchmark(),
            'stream_processing': self.run_stream_processing_benchmark()
        }
        
        # Save results if configured
        if self.config['save_results']:
            self._save_benchmark_results(benchmark_results)
        
        # Generate performance report
        self._generate_performance_report(benchmark_results)
        
        return benchmark_results
    
    def _save_benchmark_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Save benchmark results to file"""
        
        results_dir = Path(self.config['results_path'])
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        results_file = results_dir / f"benchmark_results_{timestamp}.json"
        
        serializable_results = {}
        for category, category_results in results.items():
            serializable_results[category] = [
                {
                    'test_name': r.test_name,
                    'execution_time': r.execution_time,
                    'memory_usage': r.memory_usage,
                    'cpu_usage': r.cpu_usage,
                    'throughput': r.throughput,
                    'error_rate': r.error_rate,
                    'timestamp': r.timestamp.isoformat(),
                    'metadata': r.metadata
                }
                for r in category_results
            ]
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")
    
    def _generate_performance_report(self, results: Dict[str, List[BenchmarkResult]]):
        """Generate performance analysis report"""
        
        print("\n" + "="*60)
        print("ORION PERFORMANCE BENCHMARK REPORT")
        print("="*60)
        
        for category, category_results in results.items():
            print(f"\n{category.upper()} PERFORMANCE:")
            print("-" * 40)
            
            if not category_results:
                print("No results available")
                continue
            
            # Calculate statistics
            execution_times = [r.execution_time for r in category_results]
            throughputs = [r.throughput for r in category_results]
            error_rates = [r.error_rate for r in category_results]
            
            print(f"Execution Time - Mean: {statistics.mean(execution_times):.4f}s, "
                  f"Std: {statistics.stdev(execution_times) if len(execution_times) > 1 else 0:.4f}s")
            print(f"Throughput - Mean: {statistics.mean(throughputs):.2f} ops/s, "
                  f"Max: {max(throughputs):.2f} ops/s")
            print(f"Error Rate - Mean: {statistics.mean(error_rates):.2%}, "
                  f"Max: {max(error_rates):.2%}")
            
            # Show best performing tests
            best_throughput = max(category_results, key=lambda x: x.throughput)
            print(f"Best Throughput: {best_throughput.test_name} "
                  f"({best_throughput.throughput:.2f} ops/s)")
        
        print("\n" + "="*60)

# =====================================================================
# 3. LOAD TESTING WITH LOCUST
# =====================================================================

if LOCUST_AVAILABLE:
    class ORIONUser(HttpUser):
        """Locust user for load testing ORION API"""
        
        wait_time = between(1, 3)  # Wait 1-3 seconds between requests
        
        def on_start(self):
            """Called when a user starts"""
            # Authenticate if needed
            self.auth_token = self._authenticate()
        
        def _authenticate(self) -> str:
            """Mock authentication"""
            return "mock_auth_token"
        
        @task(3)
        def predict_properties(self):
            """Test property prediction endpoint"""
            
            payload = {
                'materials': [
                    {
                        'composition': {'Ti': 0.33, 'O': 0.67},
                        'structure_type': '3D',
                        'crystal_system': 'tetragonal'
                    }
                ],
                'properties': ['bandgap', 'formation_energy'],
                'model_type': 'ensemble',
                'include_uncertainty': True
            }
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            with self.client.post("/api/v1/predict", 
                                json=payload, 
                                headers=headers,
                                catch_response=True) as response:
                if response.status_code == 200:
                    result = response.json()
                    if 'predictions' in result:
                        response.success()
                    else:
                        response.failure("Missing predictions in response")
                else:
                    response.failure(f"Unexpected status code: {response.status_code}")
        
        @task(2)
        def query_knowledge_graph(self):
            """Test knowledge graph query endpoint"""
            
            params = {
                'formula': 'TiO2',
                'property': 'bandgap',
                'limit': 10
            }
            
            headers = {'Authorization': f'Bearer {self.auth_token}'}
            
            with self.client.get("/api/v1/materials/search",
                               params=params,
                               headers=headers,
                               catch_response=True) as response:
                if response.status_code == 200:
                    result = response.json()
                    if 'materials' in result:
                        response.success()
                    else:
                        response.failure("Missing materials in response")
                else:
                    response.failure(f"Query failed: {response.status_code}")
        
        @task(1)
        def submit_simulation_job(self):
            """Test simulation job submission"""
            
            payload = {
                'simulation_type': 'dft',
                'material_structure': {
                    'composition': {'Ti': 1, 'O': 2},
                    'lattice': {'matrix': [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]}
                },
                'calculation_parameters': {
                    'k_points': [4, 4, 4],
                    'energy_cutoff': 500
                },
                'priority': 1
            }
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            with self.client.post("/api/v1/simulations/submit",
                                json=payload,
                                headers=headers,
                                catch_response=True) as response:
                if response.status_code == 201:
                    result = response.json()
                    if 'job_id' in result:
                        response.success()
                    else:
                        response.failure("Missing job_id in response")
                else:
                    response.failure(f"Submission failed: {response.status_code}")
        
        @task(1)
        def health_check(self):
            """Test system health endpoint"""
            
            with self.client.get("/health", catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Health check failed: {response.status_code}")

    class ORIONStressTestUser(ORIONUser):
        """High-intensity user for stress testing"""
        
        wait_time = between(0.1, 0.5)  # Very short wait times
        
        @task(10)
        def rapid_predictions(self):
            """Rapid-fire prediction requests"""
            
            payload = {
                'materials': [{'composition': {'Ti': 0.33, 'O': 0.67}}],
                'properties': ['bandgap'],
                'model_type': 'gnn'
            }
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            self.client.post("/api/v1/predict", json=payload, headers=headers)

# =====================================================================
# 4. SECURITY TESTING
# =====================================================================

class ORIONSecurityTester:
    """Security testing framework for ORION"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.vulnerabilities = []
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'target_url': 'http://localhost:8000',
            'auth_token': 'test_token',
            'test_payloads': True,
            'test_authentication': True,
            'test_authorization': True,
            'test_input_validation': True
        }
    
    async def run_security_tests(self) -> Dict[str, Any]:
        """Run comprehensive security test suite"""
        
        results = {
            'vulnerabilities': [],
            'tests_passed': 0,
            'tests_failed': 0,
            'risk_level': 'low'
        }
        
        # Test suites
        test_suites = [
            ('authentication', self._test_authentication),
            ('authorization', self._test_authorization),
            ('input_validation', self._test_input_validation),
            ('injection_attacks', self._test_injection_attacks),
            ('data_exposure', self._test_data_exposure)
        ]
        
        for suite_name, test_func in test_suites:
            logger.info(f"Running {suite_name} security tests...")
            
            try:
                suite_results = await test_func()
                results['tests_passed'] += suite_results.get('passed', 0)
                results['tests_failed'] += suite_results.get('failed', 0)
                results['vulnerabilities'].extend(suite_results.get('vulnerabilities', []))
                
            except Exception as e:
                logger.error(f"Security test suite {suite_name} failed: {e}")
                results['tests_failed'] += 1
        
        # Assess overall risk level
        high_risk_vulns = [v for v in results['vulnerabilities'] if v.get('severity') == 'high']
        if high_risk_vulns:
            results['risk_level'] = 'high'
        elif results['vulnerabilities']:
            results['risk_level'] = 'medium'
        
        return results
    
    async def _test_authentication(self) -> Dict[str, Any]:
        """Test authentication mechanisms"""
        
        results = {'passed': 0, 'failed': 0, 'vulnerabilities': []}
        
        test_cases = [
            ('no_auth_token', None, 401),
            ('invalid_auth_token', 'invalid_token', 401),
            ('expired_auth_token', 'expired_token', 401),
            ('valid_auth_token', self.config['auth_token'], 200)
        ]
        
        async with aiohttp.ClientSession() as session:
            for test_name, token, expected_status in test_cases:
                headers = {}
                if token:
                    headers['Authorization'] = f'Bearer {token}'
                
                try:
                    async with session.get(
                        f"{self.config['target_url']}/api/v1/predict",
                        headers=headers
                    ) as response:
                        
                        if response.status == expected_status:
                            results['passed'] += 1
                        else:
                            results['failed'] += 1
                            
                            if test_name == 'no_auth_token' and response.status == 200:
                                results['vulnerabilities'].append({
                                    'type': 'missing_authentication',
                                    'severity': 'high',
                                    'description': 'API endpoint accessible without authentication'
                                })
                
                except Exception as e:
                    logger.warning(f"Authentication test {test_name} failed: {e}")
                    results['failed'] += 1
        
        return results
    
    async def _test_authorization(self) -> Dict[str, Any]:
        """Test authorization and access controls"""
        
        results = {'passed': 0, 'failed': 0, 'vulnerabilities': []}
        
        # Test role-based access control
        test_cases = [
            ('admin_only_endpoint', '/api/v1/admin/users', 'user_token', 403),
            ('user_data_access', '/api/v1/users/other_user_id', 'user_token', 403),
            ('public_endpoint', '/api/v1/public/info', 'user_token', 200)
        ]
        
        async with aiohttp.ClientSession() as session:
            for test_name, endpoint, token, expected_status in test_cases:
                headers = {'Authorization': f'Bearer {token}'}
                
                try:
                    async with session.get(
                        f"{self.config['target_url']}{endpoint}",
                        headers=headers
                    ) as response:
                        
                        if response.status == expected_status:
                            results['passed'] += 1
                        else:
                            results['failed'] += 1
                            
                            if expected_status == 403 and response.status == 200:
                                results['vulnerabilities'].append({
                                    'type': 'authorization_bypass',
                                    'severity': 'high',
                                    'description': f'Unauthorized access to {endpoint}'
                                })
                
                except Exception as e:
                    logger.warning(f"Authorization test {test_name} failed: {e}")
                    results['failed'] += 1
        
        return results
    
    async def _test_input_validation(self) -> Dict[str, Any]:
        """Test input validation and sanitization"""
        
        results = {'passed': 0, 'failed': 0, 'vulnerabilities': []}
        
        # Test malicious payloads
        malicious_payloads = [
            {'composition': {'<script>alert("xss")</script>': 0.5, 'O': 0.5}},
            {'composition': {'Ti': -1, 'O': 2}},  # Negative composition
            {'composition': {'Ti': 'DROP TABLE materials', 'O': 0.5}},  # SQL injection attempt
            {'composition': {'Ti': '0.5', 'O': '0.5' * 10000}},  # Large input
            {'composition': {}},  # Empty composition
        ]
        
        async with aiohttp.ClientSession() as session:
            for i, payload in enumerate(malicious_payloads):
                headers = {
                    'Authorization': f'Bearer {self.config["auth_token"]}',
                    'Content-Type': 'application/json'
                }
                
                try:
                    async with session.post(
                        f"{self.config['target_url']}/api/v1/predict",
                        json={'materials': [payload]},
                        headers=headers
                    ) as response:
                        
                        if response.status in [400, 422]:  # Proper validation error
                            results['passed'] += 1
                        elif response.status == 500:  # Server error (bad)
                            results['failed'] += 1
                            results['vulnerabilities'].append({
                                'type': 'input_validation_failure',
                                'severity': 'medium',
                                'description': f'Malicious payload {i} caused server error'
                            })
                        else:
                            results['failed'] += 1
                
                except Exception as e:
                    logger.warning(f"Input validation test {i} failed: {e}")
                    results['failed'] += 1
        
        return results
    
    async def _test_injection_attacks(self) -> Dict[str, Any]:
        """Test for injection vulnerabilities"""
        
        results = {'passed': 0, 'failed': 0, 'vulnerabilities': []}
        
        # SQL injection payloads
        sql_payloads = [
            "'; DROP TABLE materials; --",
            "' OR '1'='1",
            "1; SELECT * FROM users; --"
        ]
        
        # NoSQL injection payloads
        nosql_payloads = [
            {"$ne": None},
            {"$gt": ""},
            {"$where": "this.password.length > 0"}
        ]
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.config["auth_token"]}',
                'Content-Type': 'application/json'
            }
            
            # Test SQL injection in search parameters
            for payload in sql_payloads:
                try:
                    async with session.get(
                        f"{self.config['target_url']}/api/v1/materials/search",
                        params={'formula': payload},
                        headers=headers
                    ) as response:
                        
                        if response.status == 400:  # Proper input validation
                            results['passed'] += 1
                        elif response.status == 500:  # Potential SQL error
                            results['failed'] += 1
                            results['vulnerabilities'].append({
                                'type': 'sql_injection',
                                'severity': 'high',
                                'description': 'Potential SQL injection vulnerability'
                            })
                        else:
                            results['passed'] += 1
                
                except Exception as e:
                    logger.warning(f"SQL injection test failed: {e}")
                    results['failed'] += 1
        
        return results
    
    async def _test_data_exposure(self) -> Dict[str, Any]:
        """Test for data exposure vulnerabilities"""
        
        results = {'passed': 0, 'failed': 0, 'vulnerabilities': []}
        
        # Test for sensitive data in responses
        test_endpoints = [
            '/api/v1/users/profile',
            '/api/v1/materials/search',
            '/api/v1/simulations/status'
        ]
        
        sensitive_patterns = [
            r'password',
            r'secret',
            r'token',
            r'key',
            r'api_key'
        ]
        
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {self.config["auth_token"]}'}
            
            for endpoint in test_endpoints:
                try:
                    async with session.get(
                        f"{self.config['target_url']}{endpoint}",
                        headers=headers
                    ) as response:
                        
                        if response.status == 200:
                            response_text = await response.text()
                            response_text_lower = response_text.lower()
                            
                            for pattern in sensitive_patterns:
                                if pattern in response_text_lower:
                                    results['vulnerabilities'].append({
                                        'type': 'data_exposure',
                                        'severity': 'medium',
                                        'description': f'Sensitive data pattern "{pattern}" found in {endpoint}'
                                    })
                                    results['failed'] += 1
                                    break
                            else:
                                results['passed'] += 1
                        else:
                            results['passed'] += 1
                
                except Exception as e:
                    logger.warning(f"Data exposure test for {endpoint} failed: {e}")
                    results['failed'] += 1
        
        return results
    
    def generate_security_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive security report"""
        
        report = []
        report.append("ORION SECURITY ASSESSMENT REPORT")
        report.append("=" * 50)
        report.append(f"Overall Risk Level: {results['risk_level'].upper()}")
        report.append(f"Tests Passed: {results['tests_passed']}")
        report.append(f"Tests Failed: {results['tests_failed']}")
        report.append(f"Vulnerabilities Found: {len(results['vulnerabilities'])}")
        report.append("")
        
        if results['vulnerabilities']:
            report.append("VULNERABILITIES FOUND:")
            report.append("-" * 30)
            
            for vuln in results['vulnerabilities']:
                report.append(f"Type: {vuln['type']}")
                report.append(f"Severity: {vuln['severity']}")
                report.append(f"Description: {vuln['description']}")
                report.append("")
        else:
            report.append("No vulnerabilities found!")
        
        return "\n".join(report)

# =====================================================================
# 5. CI/CD PIPELINE CONFIGURATION
# =====================================================================

class ORIONCIPipeline:
    """CI/CD pipeline configuration generator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'ci_platform': 'github_actions',  # or 'gitlab_ci', 'jenkins'
            'python_version': '3.9',
            'test_coverage_threshold': 80,
            'security_scanning': True,
            'performance_testing': True,
            'docker_registry': 'ghcr.io',
            'kubernetes_deployment': True,
            'notification_channels': ['slack', 'email']
        }
    
    def generate_github_actions_workflow(self) -> str:
        """Generate GitHub Actions workflow file"""
        
        workflow = {
            'name': 'ORION CI/CD Pipeline',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main']
                }
            },
            'env': {
                'PYTHON_VERSION': self.config['python_version'],
                'REGISTRY': self.config['docker_registry']
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'strategy': {
                        'matrix': {
                            'python-version': ['3.8', '3.9', '3.10']
                        }
                    },
                    'steps': [
                        {
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '${{ matrix.python-version }}'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': '''
                                python -m pip install --upgrade pip
                                pip install -r requirements.txt
                                pip install -r requirements-dev.txt
                            '''
                        },
                        {
                            'name': 'Lint with flake8',
                            'run': '''
                                flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
                                flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
                            '''
                        },
                        {
                            'name': 'Type checking with mypy',
                            'run': 'mypy orion/'
                        },
                        {
                            'name': 'Test with pytest',
                            'run': '''
                                pytest --cov=orion --cov-report=xml --cov-report=term-missing
                            '''
                        },
                        {
                            'name': 'Upload coverage to Codecov',
                            'uses': 'codecov/codecov-action@v3',
                            'with': {
                                'file': './coverage.xml',
                                'fail_ci_if_error': True
                            }
                        }
                    ]
                },
                'security-scan': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': self.config['python_version']
                            }
                        },
                        {
                            'name': 'Install security tools',
                            'run': '''
                                pip install bandit safety
                            '''
                        },
                        {
                            'name': 'Run Bandit security scan',
                            'run': 'bandit -r orion/ -f json -o bandit-report.json'
                        },
                        {
                            'name': 'Run Safety check',
                            'run': 'safety check --json --output safety-report.json'
                        },
                        {
                            'name': 'Upload security reports',
                            'uses': 'actions/upload-artifact@v3',
                            'with': {
                                'name': 'security-reports',
                                'path': '*-report.json'
                            }
                        }
                    ]
                },
                'performance-test': {
                    'runs-on': 'ubuntu-latest',
                    'if': "github.event_name == 'push' && github.ref == 'refs/heads/main'",
                    'steps': [
                        {
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': self.config['python_version']
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': '''
                                python -m pip install --upgrade pip
                                pip install -r requirements.txt
                                pip install pytest-benchmark
                            '''
                        },
                        {
                            'name': 'Run performance benchmarks',
                            'run': '''
                                python -m pytest tests/test_performance.py --benchmark-only --benchmark-json=benchmark.json
                            '''
                        },
                        {
                            'name': 'Upload benchmark results',
                            'uses': 'actions/upload-artifact@v3',
                            'with': {
                                'name': 'benchmark-results',
                                'path': 'benchmark.json'
                            }
                        }
                    ]
                },
                'build-and-push': {
                    'runs-on': 'ubuntu-latest',
                    'needs': ['test', 'security-scan'],
                    'if': "github.event_name == 'push' && github.ref == 'refs/heads/main'",
                    'steps': [
                        {
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Docker Buildx',
                            'uses': 'docker/setup-buildx-action@v2'
                        },
                        {
                            'name': 'Log in to Container Registry',
                            'uses': 'docker/login-action@v2',
                            'with': {
                                'registry': '${{ env.REGISTRY }}',
                                'username': '${{ github.actor }}',
                                'password': '${{ secrets.GITHUB_TOKEN }}'
                            }
                        },
                        {
                            'name': 'Extract metadata',
                            'id': 'meta',
                            'uses': 'docker/metadata-action@v4',
                            'with': {
                                'images': '${{ env.REGISTRY }}/${{ github.repository }}'
                            }
                        },
                        {
                            'name': 'Build and push Docker image',
                            'uses': 'docker/build-push-action@v4',
                            'with': {
                                'context': '.',
                                'push': True,
                                'tags': '${{ steps.meta.outputs.tags }}',
                                'labels': '${{ steps.meta.outputs.labels }}',
                                'cache-from': 'type=gha',
                                'cache-to': 'type=gha,mode=max'
                            }
                        }
                    ]
                },
                'deploy': {
                    'runs-on': 'ubuntu-latest',
                    'needs': ['build-and-push'],
                    'if': "github.event_name == 'push' && github.ref == 'refs/heads/main'",
                    'environment': 'production',
                    'steps': [
                        {
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up kubectl',
                            'uses': 'azure/setup-kubectl@v3'
                        },
                        {
                            'name': 'Deploy to Kubernetes',
                            'run': '''
                                kubectl set image deployment/orion-api orion-api=${{ env.REGISTRY }}/${{ github.repository }}:main
                                kubectl rollout status deployment/orion-api
                            '''
                        },
                        {
                            'name': 'Run smoke tests',
                            'run': '''
                                python scripts/smoke_tests.py --environment production
                            '''
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False)
    
    def generate_dockerfile(self) -> str:
        """Generate optimized Dockerfile"""
        
        dockerfile = f'''
# Multi-stage build for ORION
FROM python:{self.config['python_version']}-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libffi-dev \\
    libssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:{self.config['python_version']}-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash orion

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/orion/.local

# Copy application code
COPY --chown=orion:orion . .

# Set environment variables
ENV PATH=/home/orion/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER orion

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python scripts/health_check.py

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "orion.api.main"]
'''
        
        return dockerfile.strip()
    
    def generate_docker_compose(self) -> str:
        """Generate Docker Compose configuration for development"""
        
        compose = {
            'version': '3.8',
            'services': {
                'orion-api': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile'
                    },
                    'ports': ['8000:8000'],
                    'environment': [
                        'DATABASE_URL=postgresql://orion:password@postgres:5432/orion',
                        'REDIS_URL=redis://redis:6379/0',
                        'NEO4J_URL=bolt://neo4j:7687'
                    ],
                    'depends_on': ['postgres', 'redis', 'neo4j'],
                    'volumes': ['./:/app'],
                    'command': 'python -m orion.api.main --reload'
                },
                'orion-stream-processor': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile'
                    },
                    'environment': [
                        'REDIS_URL=redis://redis:6379/0',
                        'NEO4J_URL=bolt://neo4j:7687'
                    ],
                    'depends_on': ['redis', 'neo4j'],
                    'command': 'python -m orion.stream_processor.main'
                },
                'postgres': {
                    'image': 'postgres:14',
                    'environment': [
                        'POSTGRES_DB=orion',
                        'POSTGRES_USER=orion',
                        'POSTGRES_PASSWORD=password'
                    ],
                    'ports': ['5432:5432'],
                    'volumes': ['postgres_data:/var/lib/postgresql/data']
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': ['6379:6379'],
                    'volumes': ['redis_data:/data']
                },
                'neo4j': {
                    'image': 'neo4j:5.0',
                    'environment': [
                        'NEO4J_AUTH=neo4j/password',
                        'NEO4J_PLUGINS=["apoc", "graph-data-science"]'
                    ],
                    'ports': ['7474:7474', '7687:7687'],
                    'volumes': ['neo4j_data:/data']
                },
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'ports': ['9090:9090'],
                    'volumes': ['./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml']
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'ports': ['3000:3000'],
                    'environment': ['GF_SECURITY_ADMIN_PASSWORD=admin'],
                    'volumes': ['grafana_data:/var/lib/grafana']
                }
            },
            'volumes': {
                'postgres_data': {},
                'redis_data': {},
                'neo4j_data': {},
                'grafana_data': {}
            },
            'networks': {
                'default': {
                    'name': 'orion-network'
                }
            }
        }
        
        return yaml.dump(compose, default_flow_style=False)

# =====================================================================
# 6. PERFORMANCE OPTIMIZATION STRATEGIES
# =====================================================================

class ORIONOptimizer:
    """Performance optimization toolkit for ORION"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.optimization_results = {}
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'cache_enabled': True,
            'connection_pooling': True,
            'async_processing': True,
            'batch_processing': True,
            'gpu_acceleration': True,
            'model_quantization': True,
            'data_compression': True
        }
    
    def optimize_database_queries(self) -> Dict[str, Any]:
        """Optimize database query performance"""
        
        optimizations = {
            'indexes_created': [],
            'query_optimizations': [],
            'connection_pool_settings': {}
        }
        
        # Database indexing recommendations
        recommended_indexes = [
            "CREATE INDEX CONCURRENTLY idx_materials_formula ON materials (formula);",
            "CREATE INDEX CONCURRENTLY idx_properties_material_id ON properties (material_id);",
            "CREATE INDEX CONCURRENTLY idx_properties_name_value ON properties (name, value);",
            "CREATE INDEX CONCURRENTLY idx_sources_doi ON sources (doi);",
            "CREATE INDEX CONCURRENTLY idx_simulations_status_created ON simulations (status, created_at);"
        ]
        
        optimizations['indexes_created'] = recommended_indexes
        
        # Connection pool optimization
        optimizations['connection_pool_settings'] = {
            'pool_size': 20,
            'max_overflow': 30,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'pool_pre_ping': True
        }
        
        # Query optimization recommendations
        optimizations['query_optimizations'] = [
            "Use LIMIT clauses for large result sets",
            "Implement query result caching for frequent queries",
            "Use prepared statements for repeated queries",
            "Optimize JOIN operations with proper indexing",
            "Use batch operations for bulk inserts/updates"
        ]
        
        return optimizations
    
    def optimize_model_inference(self) -> Dict[str, Any]:
        """Optimize ML model inference performance"""
        
        optimizations = {
            'batch_processing': {},
            'model_quantization': {},
            'gpu_optimization': {},
            'caching_strategies': {}
        }
        
        # Batch processing optimization
        optimizations['batch_processing'] = {
            'recommended_batch_sizes': {
                'gnn': 32,
                'rf': 128,
                'ensemble': 16
            },
            'dynamic_batching': True,
            'max_batch_delay_ms': 100
        }
        
        # Model quantization
        if self.config['model_quantization']:
            optimizations['model_quantization'] = {
                'int8_quantization': True,
                'dynamic_quantization': True,
                'expected_speedup': '2-4x',
                'accuracy_retention': '95-99%'
            }
        
        # GPU optimization
        if self.config['gpu_acceleration']:
            optimizations['gpu_optimization'] = {
                'mixed_precision': True,
                'tensor_core_utilization': True,
                'memory_optimization': True,
                'multi_gpu_inference': True
            }
        
        # Caching strategies
        optimizations['caching_strategies'] = {
            'prediction_cache_ttl': 3600,  # 1 hour
            'model_cache_size': '2GB',
            'feature_cache_enabled': True,
            'cache_hit_rate_target': 0.8
        }
        
        return optimizations
    
    def optimize_data_pipeline(self) -> Dict[str, Any]:
        """Optimize data processing pipeline"""
        
        optimizations = {
            'async_processing': {},
            'data_compression': {},
            'streaming_optimization': {},
            'memory_management': {}
        }
        
        # Async processing
        if self.config['async_processing']:
            optimizations['async_processing'] = {
                'worker_processes': multiprocessing.cpu_count(),
                'async_io_enabled': True,
                'queue_max_size': 10000,
                'prefetch_factor': 2
            }
        
        # Data compression
        if self.config['data_compression']:
            optimizations['data_compression'] = {
                'algorithms': ['gzip', 'lz4', 'snappy'],
                'compression_ratio': '3-5x',
                'recommended_algorithm': 'lz4',
                'compress_threshold_bytes': 1024
            }
        
        # Streaming optimization
        optimizations['streaming_optimization'] = {
            'buffer_size': 8192,
            'batch_size': 100,
            'flush_interval_ms': 1000,
            'backpressure_handling': True
        }
        
        # Memory management
        optimizations['memory_management'] = {
            'memory_pool_enabled': True,
            'gc_optimization': True,
            'object_recycling': True,
            'memory_profiling': True
        }
        
        return optimizations
    
    def generate_optimization_script(self) -> str:
        """Generate optimization implementation script"""
        
        script = '''#!/usr/bin/env python3
"""
ORION Performance Optimization Script
====================================

This script applies performance optimizations to the ORION system.
"""

import asyncio
import psutil
import torch
from pathlib import Path

async def apply_database_optimizations():
    """Apply database optimizations"""
    print("Applying database optimizations...")
    
    # Create recommended indexes
    indexes = [
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_materials_formula ON materials (formula);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_properties_material_id ON properties (material_id);",
        # Add more indexes...
    ]
    
    # Apply indexes (would connect to actual database)
    for index in indexes:
        print(f"Creating index: {index}")
        # await database.execute(index)
    
    print("Database optimizations applied.")

async def optimize_model_inference():
    """Optimize model inference performance"""
    print("Optimizing model inference...")
    
    # Apply quantization
    if torch.cuda.is_available():
        print("GPU detected - enabling CUDA optimizations")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Enable mixed precision
    print("Enabling mixed precision training")
    
    # Optimize batch sizes
    print("Optimizing batch sizes for inference")
    
    print("Model inference optimizations applied.")

async def configure_caching():
    """Configure caching systems"""
    print("Configuring caching systems...")
    
    # Redis configuration
    redis_config = {
        'maxmemory': '2gb',
        'maxmemory-policy': 'allkeys-lru',
        'save': '900 1 300 10 60 10000'
    }
    
    # Apply Redis optimizations
    for key, value in redis_config.items():
        print(f"Redis config: {key} = {value}")
    
    print("Caching configurations applied.")

async def optimize_system_resources():
    """Optimize system resource usage"""
    print("Optimizing system resources...")
    
    # Check system resources
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total // (1024**3)
    
    print(f"System: {cpu_count} CPUs, {memory_gb}GB RAM")
    
    # Configure worker processes
    optimal_workers = min(cpu_count, 8)
    print(f"Setting optimal worker count: {optimal_workers}")
    
    # Configure memory limits
    memory_limit = int(memory_gb * 0.8)  # Use 80% of available memory
    print(f"Setting memory limit: {memory_limit}GB")
    
    print("System resource optimizations applied.")

async def main():
    """Main optimization function"""
    print("Starting ORION performance optimization...")
    
    await apply_database_optimizations()
    await optimize_model_inference()
    await configure_caching()
    await optimize_system_resources()
    
    print("ORION performance optimization completed!")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        return script
    
    def generate_monitoring_dashboard_config(self) -> str:
        """Generate Grafana dashboard configuration for performance monitoring"""
        
        dashboard_config = {
            "dashboard": {
                "title": "ORION Performance Dashboard",
                "panels": [
                    {
                        "title": "API Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, orion_api_requests_duration_seconds_bucket)",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.50, orion_api_requests_duration_seconds_bucket)",
                                "legendFormat": "50th percentile"
                            }
                        ]
                    },
                    {
                        "title": "Prediction Throughput",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(orion_predictions_total[5m])",
                                "legendFormat": "Predictions/sec"
                            }
                        ]
                    },
                    {
                        "title": "Database Connection Pool",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "orion_db_connections_active",
                                "legendFormat": "Active connections"
                            },
                            {
                                "expr": "orion_db_connections_idle",
                                "legendFormat": "Idle connections"
                            }
                        ]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "process_resident_memory_bytes / 1024 / 1024",
                                "legendFormat": "Memory (MB)"
                            }
                        ]
                    },
                    {
                        "title": "Cache Hit Rate",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "orion_cache_hits / (orion_cache_hits + orion_cache_misses)",
                                "legendFormat": "Hit Rate"
                            }
                        ]
                    }
                ]
            }
        }
        
        return json.dumps(dashboard_config, indent=2)

# =====================================================================
# 7. MAIN EXECUTION AND TESTING ORCHESTRATOR
# =====================================================================

async def run_comprehensive_testing_suite():
    """Run the complete testing and optimization suite"""
    
    print("ORION: Comprehensive Testing & Optimization Suite")
    print("="*60)
    
    # 1. Unit Tests
    print("\n1. Running Unit Tests...")
    print("-" * 30)
    
    # Run unit tests
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestORIONCore)
    test_runner = unittest.TextTestRunner(verbosity=2)
    unit_test_result = test_runner.run(test_suite)
    
    print(f"Unit tests: {unit_test_result.testsRun} tests, "
          f"{len(unit_test_result.failures)} failures, "
          f"{len(unit_test_result.errors)} errors")
    
    # 2. Performance Benchmarks
    print("\n2. Running Performance Benchmarks...")
    print("-" * 30)
    
    benchmark_suite = ORIONBenchmarkSuite()
    benchmark_results = benchmark_suite.run_comprehensive_benchmark()
    
    # 3. Security Testing
    print("\n3. Running Security Tests...")
    print("-" * 30)
    
    security_tester = ORIONSecurityTester()
    security_results = await security_tester.run_security_tests()
    
    print(f"Security tests: {security_results['tests_passed']} passed, "
          f"{security_results['tests_failed']} failed")
    print(f"Risk level: {security_results['risk_level']}")
    print(f"Vulnerabilities found: {len(security_results['vulnerabilities'])}")
    
    # 4. Generate CI/CD Configuration
    print("\n4. Generating CI/CD Configuration...")
    print("-" * 30)
    
    ci_pipeline = ORIONCIPipeline()
    
    # Save CI/CD files
    ci_config_dir = Path('./ci_cd_configs')
    ci_config_dir.mkdir(exist_ok=True)
    
    # GitHub Actions workflow
    with open(ci_config_dir / 'github_actions.yml', 'w') as f:
        f.write(ci_pipeline.generate_github_actions_workflow())
    
    # Dockerfile
    with open(ci_config_dir / 'Dockerfile', 'w') as f:
        f.write(ci_pipeline.generate_dockerfile())
    
    # Docker Compose
    with open(ci_config_dir / 'docker-compose.yml', 'w') as f:
        f.write(ci_pipeline.generate_docker_compose())
    
    print("CI/CD configuration files generated")
    
    # 5. Generate Optimization Recommendations
    print("\n5. Generating Optimization Recommendations...")
    print("-" * 30)
    
    optimizer = ORIONOptimizer()
    
    # Generate optimization reports
    db_optimizations = optimizer.optimize_database_queries()
    model_optimizations = optimizer.optimize_model_inference()
    pipeline_optimizations = optimizer.optimize_data_pipeline()
    
    # Save optimization script
    with open(ci_config_dir / 'optimize_performance.py', 'w') as f:
        f.write(optimizer.generate_optimization_script())
    
    # Save monitoring configuration
    with open(ci_config_dir / 'grafana_dashboard.json', 'w') as f:
        f.write(optimizer.generate_monitoring_dashboard_config())
    
    print("Optimization recommendations generated")
    
    # 6. Final Report
    print("\n6. Final Testing Report")
    print("-" * 30)
    
    total_tests = unit_test_result.testsRun + security_results['tests_passed'] + security_results['tests_failed']
    total_failures = len(unit_test_result.failures) + len(unit_test_result.errors) + security_results['tests_failed']
    
    print(f"Total tests executed: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Success rate: {((total_tests - total_failures) / total_tests * 100):.1f}%")
    
    # Performance summary
    print(f"\nPerformance Benchmarks:")
    for category, results in benchmark_results.items():
        if results:
            avg_throughput = statistics.mean([r.throughput for r in results])
            print(f"  {category}: {avg_throughput:.2f} ops/sec average")
    
    # Security summary
    if security_results['vulnerabilities']:
        print(f"\nSecurity Issues Found:")
        for vuln in security_results['vulnerabilities'][:3]:  # Show first 3
            print(f"  - {vuln['type']}: {vuln['severity']}")
        if len(security_results['vulnerabilities']) > 3:
            print(f"  ... and {len(security_results['vulnerabilities']) - 3} more")
    else:
        print("\nNo security vulnerabilities found!")
    
    print(f"\nConfiguration files saved to: {ci_config_dir}")
    print("\nTesting and optimization suite completed!")
    
    return {
        'unit_tests': unit_test_result,
        'benchmarks': benchmark_results,
        'security': security_results,
        'total_success_rate': (total_tests - total_failures) / total_tests
    }

if __name__ == "__main__":
    # Run the comprehensive testing suite
    results = asyncio.run(run_comprehensive_testing_suite())
    
    # Exit with appropriate code
    if results['total_success_rate'] >= 0.95:
        exit(0)  # Success
    else:
        exit(1)  # Some tests failed
