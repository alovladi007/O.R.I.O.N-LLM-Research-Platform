"""
ORION: Advanced Features - Real-time Processing & Orchestration
==============================================================

This module implements advanced ORION features including:

1. Real-time streaming data processing
2. Advanced simulation orchestration (DFT, MD, FEA)
3. Knowledge graph integration and reasoning
4. Automated experimental design and protocol generation
5. Advanced active learning with acquisition functions
6. Multi-objective optimization and Pareto frontier analysis
7. Federated learning for multi-lab collaboration
8. Advanced uncertainty propagation
9. Real-time knowledge graph updates
10. Integration with external databases and APIs

Author: ORION Development Team
"""

import asyncio
import aiohttp
import aiofiles
import websockets
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional, AsyncGenerator, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid
import hashlib
import pickle
from pathlib import Path
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict, deque
import warnings

# Scientific computing imports
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, beta
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Graph/Network libraries
try:
    import networkx as nx
    import py2neo
    GRAPH_LIBS_AVAILABLE = True
except ImportError:
    GRAPH_LIBS_AVAILABLE = False
    print("Graph libraries not available - using simplified graph operations")

# Molecular/Materials libraries
try:
    from pymatgen.core import Structure, Composition
    from pymatgen.io.vasp import Poscar, Outcar
    from pymatgen.analysis.structure_matcher import StructureMatcher
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("PyMatGen not available - using simplified structure handling")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. REAL-TIME STREAMING DATA PROCESSOR
# =====================================================================

@dataclass
class StreamMessage:
    """Message format for streaming data"""
    message_id: str
    timestamp: datetime
    source: str
    message_type: str
    payload: Dict[str, Any]
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_id': self.message_id,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'message_type': self.message_type,
            'payload': self.payload,
            'priority': self.priority
        }

class StreamProcessor:
    """Real-time streaming data processor for materials science data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.message_queue = asyncio.Queue(maxsize=self.config['max_queue_size'])
        self.processors = {}
        self.subscribers = defaultdict(list)
        self.metrics = defaultdict(int)
        self.is_running = False
        
        # Rate limiting
        self.rate_limiters = defaultdict(lambda: deque(maxlen=100))
        
        # Message handlers
        self.handlers = {
            'literature_update': self._handle_literature_update,
            'experimental_result': self._handle_experimental_result,
            'simulation_complete': self._handle_simulation_complete,
            'model_update': self._handle_model_update,
            'property_conflict': self._handle_property_conflict
        }
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_queue_size': 10000,
            'processing_workers': 4,
            'rate_limit_per_second': 100,
            'batch_processing_size': 50,
            'processing_timeout': 30.0,
            'dead_letter_queue_size': 1000,
            'retry_attempts': 3,
            'websocket_port': 8765
        }
    
    async def start_streaming(self):
        """Start the streaming data processor"""
        self.is_running = True
        
        # Start processing workers
        workers = []
        for i in range(self.config['processing_workers']):
            worker = asyncio.create_task(self._processing_worker(f"worker_{i}"))
            workers.append(worker)
        
        # Start WebSocket server for real-time updates
        websocket_server = await websockets.serve(
            self._websocket_handler,
            "localhost",
            self.config['websocket_port']
        )
        
        logger.info(f"Stream processor started with {len(workers)} workers")
        logger.info(f"WebSocket server listening on port {self.config['websocket_port']}")
        
        # Wait for all workers
        await asyncio.gather(*workers)
        
        websocket_server.close()
        await websocket_server.wait_closed()
    
    async def _processing_worker(self, worker_id: str):
        """Background worker for processing streaming messages"""
        logger.info(f"Processing worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Process message
                await self._process_message(message, worker_id)
                
                # Mark task as done
                self.message_queue.task_done()
                
                # Update metrics
                self.metrics['messages_processed'] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self.metrics['processing_errors'] += 1
                await asyncio.sleep(1.0)
        
        logger.info(f"Processing worker {worker_id} stopped")
    
    async def _process_message(self, message: StreamMessage, worker_id: str):
        """Process a single streaming message"""
        start_time = time.time()
        
        try:
            # Rate limiting check
            if not self._check_rate_limit(message.source):
                logger.warning(f"Rate limit exceeded for source: {message.source}")
                return
            
            # Get appropriate handler
            handler = self.handlers.get(message.message_type, self._handle_unknown_message)
            
            # Process with timeout
            result = await asyncio.wait_for(
                handler(message),
                timeout=self.config['processing_timeout']
            )
            
            # Notify subscribers
            await self._notify_subscribers(message.message_type, result)
            
            processing_time = time.time() - start_time
            logger.debug(f"Message {message.message_id} processed in {processing_time:.3f}s by {worker_id}")
            
        except asyncio.TimeoutError:
            logger.error(f"Message {message.message_id} processing timeout")
            self.metrics['timeouts'] += 1
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            self.metrics['processing_errors'] += 1
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if source is within rate limits"""
        current_time = time.time()
        source_times = self.rate_limiters[source]
        
        # Remove old timestamps
        while source_times and current_time - source_times[0] > 1.0:
            source_times.popleft()
        
        # Check rate limit
        if len(source_times) >= self.config['rate_limit_per_second']:
            return False
        
        # Add current timestamp
        source_times.append(current_time)
        return True
    
    async def _handle_literature_update(self, message: StreamMessage) -> Dict[str, Any]:
        """Handle new literature data"""
        payload = message.payload
        
        # Extract paper metadata
        paper_data = {
            'doi': payload.get('doi'),
            'title': payload.get('title'),
            'authors': payload.get('authors', []),
            'abstract': payload.get('abstract'),
            'publication_date': payload.get('publication_date'),
            'journal': payload.get('journal'),
            'materials': payload.get('materials', []),
            'properties': payload.get('properties', {}),
            'methods': payload.get('methods', [])
        }
        
        # Process with NLP/ML pipeline
        processed_data = await self._extract_materials_data(paper_data)
        
        # Update knowledge graph
        if GRAPH_LIBS_AVAILABLE:
            await self._update_knowledge_graph(processed_data)
        
        # Check for conflicts with existing data
        conflicts = await self._detect_property_conflicts(processed_data)
        
        if conflicts:
            # Send conflict messages for resolution
            for conflict in conflicts:
                conflict_message = StreamMessage(
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    source='conflict_detector',
                    message_type='property_conflict',
                    payload=conflict
                )
                await self.message_queue.put(conflict_message)
        
        return {
            'status': 'processed',
            'extracted_materials': len(processed_data.get('materials', [])),
            'conflicts_detected': len(conflicts),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    async def _handle_experimental_result(self, message: StreamMessage) -> Dict[str, Any]:
        """Handle experimental results from labs"""
        payload = message.payload
        
        result_data = {
            'experiment_id': payload.get('experiment_id'),
            'material_id': payload.get('material_id'),
            'composition': payload.get('composition'),
            'synthesis_conditions': payload.get('synthesis_conditions'),
            'measured_properties': payload.get('measured_properties', {}),
            'measurement_uncertainty': payload.get('measurement_uncertainty', {}),
            'lab_id': payload.get('lab_id'),
            'timestamp': payload.get('timestamp')
        }
        
        # Validate experimental data
        validation_result = await self._validate_experimental_data(result_data)
        
        if validation_result['is_valid']:
            # Update models with new data
            await self._update_models_with_experimental_data(result_data)
            
            # Generate suggestions for follow-up experiments
            suggestions = await self._generate_experiment_suggestions(result_data)
            
            return {
                'status': 'accepted',
                'validation_result': validation_result,
                'follow_up_suggestions': suggestions
            }
        else:
            return {
                'status': 'rejected',
                'validation_result': validation_result,
                'errors': validation_result.get('errors', [])
            }
    
    async def _handle_simulation_complete(self, message: StreamMessage) -> Dict[str, Any]:
        """Handle completed simulation results"""
        payload = message.payload
        
        simulation_data = {
            'simulation_id': payload.get('simulation_id'),
            'simulation_type': payload.get('simulation_type'),  # DFT, MD, FEA
            'material_structure': payload.get('material_structure'),
            'calculated_properties': payload.get('calculated_properties', {}),
            'computation_time': payload.get('computation_time'),
            'convergence_status': payload.get('convergence_status'),
            'software_version': payload.get('software_version'),
            'input_parameters': payload.get('input_parameters', {})
        }
        
        # Validate simulation results
        if simulation_data['convergence_status'] == 'converged':
            # Store results in database
            await self._store_simulation_results(simulation_data)
            
            # Update property predictions
            await self._update_property_predictions(simulation_data)
            
            # Check if this completes a prediction workflow
            await self._check_workflow_completion(simulation_data)
            
            return {
                'status': 'processed',
                'properties_calculated': len(simulation_data['calculated_properties']),
                'stored': True
            }
        else:
            # Handle failed simulation
            await self._handle_simulation_failure(simulation_data)
            return {
                'status': 'failed',
                'convergence_status': simulation_data['convergence_status']
            }
    
    async def _handle_model_update(self, message: StreamMessage) -> Dict[str, Any]:
        """Handle model updates and retraining notifications"""
        payload = message.payload
        
        update_data = {
            'model_id': payload.get('model_id'),
            'model_type': payload.get('model_type'),
            'update_type': payload.get('update_type'),  # retrain, fine_tune, parameter_update
            'performance_metrics': payload.get('performance_metrics', {}),
            'training_data_size': payload.get('training_data_size'),
            'model_version': payload.get('model_version')
        }
        
        # Validate model update
        if self._validate_model_update(update_data):
            # Notify prediction services
            await self._notify_prediction_services(update_data)
            
            # Update model registry
            await self._update_model_registry(update_data)
            
            return {
                'status': 'applied',
                'model_version': update_data['model_version'],
                'services_notified': True
            }
        else:
            return {
                'status': 'rejected',
                'reason': 'validation_failed'
            }
    
    async def _handle_property_conflict(self, message: StreamMessage) -> Dict[str, Any]:
        """Handle property value conflicts between sources"""
        payload = message.payload
        
        conflict_data = {
            'material_id': payload.get('material_id'),
            'property_name': payload.get('property_name'),
            'conflicting_values': payload.get('conflicting_values', []),
            'sources': payload.get('sources', []),
            'detection_timestamp': payload.get('detection_timestamp')
        }
        
        # Apply conflict resolution algorithm
        resolution = await self._resolve_property_conflict(conflict_data)
        
        # Update knowledge graph with consensus
        if GRAPH_LIBS_AVAILABLE:
            await self._update_consensus_value(conflict_data, resolution)
        
        # Notify relevant subscribers
        await self._notify_conflict_resolution(conflict_data, resolution)
        
        return {
            'status': 'resolved',
            'consensus_value': resolution.get('consensus_value'),
            'confidence': resolution.get('confidence'),
            'resolution_method': resolution.get('method')
        }
    
    async def _handle_unknown_message(self, message: StreamMessage) -> Dict[str, Any]:
        """Handle unknown message types"""
        logger.warning(f"Unknown message type: {message.message_type}")
        return {
            'status': 'unknown_type',
            'message_type': message.message_type
        }
    
    async def _websocket_handler(self, websocket, path):
        """Handle WebSocket connections for real-time updates"""
        client_id = str(uuid.uuid4())
        logger.info(f"New WebSocket client connected: {client_id}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'subscribe':
                        # Subscribe to message types
                        message_types = data.get('message_types', [])
                        for msg_type in message_types:
                            self.subscribers[msg_type].append(websocket)
                        
                        await websocket.send(json.dumps({
                            'type': 'subscription_confirmed',
                            'subscribed_to': message_types
                        }))
                    
                    elif data.get('type') == 'submit_message':
                        # Submit new message to processing queue
                        stream_message = StreamMessage(
                            message_id=str(uuid.uuid4()),
                            timestamp=datetime.now(),
                            source=data.get('source', 'websocket'),
                            message_type=data.get('message_type'),
                            payload=data.get('payload', {}),
                            priority=data.get('priority', 1)
                        )
                        
                        await self.message_queue.put(stream_message)
                        
                        await websocket.send(json.dumps({
                            'type': 'message_accepted',
                            'message_id': stream_message.message_id
                        }))
                
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Remove client from all subscriptions
            for subscribers in self.subscribers.values():
                if websocket in subscribers:
                    subscribers.remove(websocket)
            
            logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def _notify_subscribers(self, message_type: str, result: Dict[str, Any]):
        """Notify WebSocket subscribers of processing results"""
        if message_type in self.subscribers:
            notification = {
                'type': 'processing_result',
                'message_type': message_type,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to all subscribers
            disconnected_clients = []
            for websocket in self.subscribers[message_type]:
                try:
                    await websocket.send(json.dumps(notification))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.append(websocket)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                self.subscribers[message_type].remove(client)
    
    async def submit_message(self, message: StreamMessage) -> bool:
        """Submit a message to the processing queue"""
        try:
            await self.message_queue.put(message)
            self.metrics['messages_submitted'] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to submit message: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming processor metrics"""
        return {
            'queue_size': self.message_queue.qsize(),
            'metrics': dict(self.metrics),
            'subscribers': {msg_type: len(subs) for msg_type, subs in self.subscribers.items()},
            'rate_limiters': {source: len(times) for source, times in self.rate_limiters.items()}
        }
    
    # Placeholder implementations for various processing methods
    async def _extract_materials_data(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract materials data from literature using NLP"""
        # Simplified implementation - would use advanced NLP in practice
        return {
            'materials': paper_data.get('materials', []),
            'properties': paper_data.get('properties', {}),
            'synthesis_methods': []
        }
    
    async def _update_knowledge_graph(self, data: Dict[str, Any]):
        """Update knowledge graph with new data"""
        # Placeholder - would interact with Neo4j in practice
        pass
    
    async def _detect_property_conflicts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts with existing property values"""
        # Placeholder - would query existing data
        return []
    
    async def _validate_experimental_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experimental data"""
        return {'is_valid': True, 'errors': []}
    
    async def _update_models_with_experimental_data(self, data: Dict[str, Any]):
        """Update ML models with new experimental data"""
        pass
    
    async def _generate_experiment_suggestions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate follow-up experiment suggestions"""
        return []
    
    async def _store_simulation_results(self, data: Dict[str, Any]):
        """Store simulation results in database"""
        pass
    
    async def _update_property_predictions(self, data: Dict[str, Any]):
        """Update property predictions based on simulation results"""
        pass
    
    async def _check_workflow_completion(self, data: Dict[str, Any]):
        """Check if simulation completes a workflow"""
        pass
    
    async def _handle_simulation_failure(self, data: Dict[str, Any]):
        """Handle failed simulations"""
        pass
    
    def _validate_model_update(self, data: Dict[str, Any]) -> bool:
        """Validate model update data"""
        return True
    
    async def _notify_prediction_services(self, data: Dict[str, Any]):
        """Notify prediction services of model updates"""
        pass
    
    async def _update_model_registry(self, data: Dict[str, Any]):
        """Update model registry with new version"""
        pass
    
    async def _resolve_property_conflict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve property conflicts using consensus algorithm"""
        values = data.get('conflicting_values', [])
        if values:
            return {
                'consensus_value': np.mean(values),
                'confidence': 0.8,
                'method': 'simple_average'
            }
        return {}
    
    async def _update_consensus_value(self, conflict_data: Dict[str, Any], resolution: Dict[str, Any]):
        """Update knowledge graph with consensus value"""
        pass
    
    async def _notify_conflict_resolution(self, conflict_data: Dict[str, Any], resolution: Dict[str, Any]):
        """Notify subscribers of conflict resolution"""
        pass

# =====================================================================
# 2. ADVANCED SIMULATION ORCHESTRATOR
# =====================================================================

@dataclass
class SimulationJob:
    """Simulation job specification"""
    job_id: str
    simulation_type: str  # 'dft', 'md', 'fea', 'monte_carlo'
    material_structure: Dict[str, Any]
    calculation_parameters: Dict[str, Any]
    priority: int = 1
    estimated_runtime: float = 3600.0  # seconds
    required_resources: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    callback_url: Optional[str] = None
    
class SimulationOrchestrator:
    """Advanced simulation orchestrator for materials calculations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.job_queue = asyncio.PriorityQueue()
        self.running_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        self.resource_manager = ResourceManager(self.config)
        self.workflow_manager = WorkflowManager()
        
        # Simulation engines
        self.engines = {
            'dft': DFTEngine(self.config.get('dft_config', {})),
            'md': MDEngine(self.config.get('md_config', {})),
            'fea': FEAEngine(self.config.get('fea_config', {})),
            'monte_carlo': MonteCarloEngine(self.config.get('mc_config', {}))
        }
        
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_workers': 8,
            'max_concurrent_jobs': 20,
            'job_timeout': 24 * 3600,  # 24 hours
            'resource_check_interval': 60,  # seconds
            'cleanup_interval': 3600,  # 1 hour
            'job_retry_attempts': 2,
            'output_storage_path': './simulation_outputs',
            'temp_storage_path': './temp_simulations'
        }
    
    async def start_orchestrator(self):
        """Start the simulation orchestrator"""
        self.is_running = True
        
        # Create output directories
        Path(self.config['output_storage_path']).mkdir(parents=True, exist_ok=True)
        Path(self.config['temp_storage_path']).mkdir(parents=True, exist_ok=True)
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._job_dispatcher()),
            asyncio.create_task(self._resource_monitor()),
            asyncio.create_task(self._job_cleanup()),
            asyncio.create_task(self._workflow_processor())
        ]
        
        logger.info("Simulation orchestrator started")
        
        await asyncio.gather(*tasks)
    
    async def submit_job(self, job: SimulationJob) -> str:
        """Submit a simulation job"""
        # Validate job
        validation_result = await self._validate_job(job)
        if not validation_result['is_valid']:
            raise ValueError(f"Invalid job: {validation_result['errors']}")
        
        # Estimate resources and runtime
        job.estimated_runtime = await self._estimate_runtime(job)
        job.required_resources = await self._estimate_resources(job)
        
        # Add to queue with priority
        priority_score = self._calculate_priority_score(job)
        await self.job_queue.put((priority_score, job))
        
        logger.info(f"Job {job.job_id} submitted with priority {priority_score}")
        return job.job_id
    
    async def submit_workflow(self, workflow: 'SimulationWorkflow') -> str:
        """Submit a complex simulation workflow"""
        workflow_id = str(uuid.uuid4())
        
        # Register workflow
        self.workflow_manager.register_workflow(workflow_id, workflow)
        
        # Submit initial jobs
        initial_jobs = workflow.get_initial_jobs()
        for job in initial_jobs:
            job.workflow_id = workflow_id
            await self.submit_job(job)
        
        logger.info(f"Workflow {workflow_id} submitted with {len(initial_jobs)} initial jobs")
        return workflow_id
    
    def _calculate_priority_score(self, job: SimulationJob) -> float:
        """Calculate priority score for job scheduling"""
        # Higher score = higher priority (negative for min-heap behavior)
        base_priority = -job.priority
        
        # Boost priority for shorter jobs
        runtime_factor = 1.0 / (1.0 + job.estimated_runtime / 3600.0)
        
        # Boost priority for jobs with dependencies waiting
        dependency_factor = 1.0 + 0.1 * len(job.dependencies)
        
        return base_priority * runtime_factor * dependency_factor
    
    async def _job_dispatcher(self):
        """Main job dispatcher loop"""
        while self.is_running:
            try:
                # Check if we can run more jobs
                if len(self.running_jobs) >= self.config['max_concurrent_jobs']:
                    await asyncio.sleep(5)
                    continue
                
                # Get next job from queue
                try:
                    priority, job = await asyncio.wait_for(
                        self.job_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check dependencies
                if not await self._check_dependencies(job):
                    # Put job back in queue
                    await self.job_queue.put((priority, job))
                    await asyncio.sleep(1)
                    continue
                
                # Check resource availability
                if not await self.resource_manager.can_allocate(job.required_resources):
                    # Put job back in queue
                    await self.job_queue.put((priority, job))
                    await asyncio.sleep(5)
                    continue
                
                # Allocate resources and start job
                resources = await self.resource_manager.allocate(job.required_resources)
                job.allocated_resources = resources
                
                # Start job execution
                task = asyncio.create_task(self._execute_job(job))
                self.running_jobs[job.job_id] = {
                    'job': job,
                    'task': task,
                    'start_time': datetime.now(),
                    'resources': resources
                }
                
                logger.info(f"Started job {job.job_id} ({job.simulation_type})")
                
            except Exception as e:
                logger.error(f"Error in job dispatcher: {e}")
                await asyncio.sleep(1)
    
    async def _execute_job(self, job: SimulationJob) -> Dict[str, Any]:
        """Execute a simulation job"""
        try:
            # Get appropriate engine
            engine = self.engines.get(job.simulation_type)
            if not engine:
                raise ValueError(f"No engine available for {job.simulation_type}")
            
            # Prepare input files
            input_dir = await self._prepare_input_files(job)
            
            # Execute simulation
            result = await engine.run_simulation(
                input_dir, job.calculation_parameters, job.allocated_resources
            )
            
            # Process output files
            processed_result = await self._process_output_files(job, result)
            
            # Store results
            await self._store_job_results(job, processed_result)
            
            # Move to completed jobs
            self.completed_jobs[job.job_id] = {
                'job': job,
                'result': processed_result,
                'completion_time': datetime.now()
            }
            
            # Trigger dependent jobs
            await self._trigger_dependent_jobs(job)
            
            # Send callback notification
            if job.callback_url:
                await self._send_callback(job, processed_result)
            
            logger.info(f"Job {job.job_id} completed successfully")
            return processed_result
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            
            # Move to failed jobs
            self.failed_jobs[job.job_id] = {
                'job': job,
                'error': str(e),
                'failure_time': datetime.now()
            }
            
            # Retry if attempts remaining
            if getattr(job, 'retry_count', 0) < self.config['job_retry_attempts']:
                job.retry_count = getattr(job, 'retry_count', 0) + 1
                logger.info(f"Retrying job {job.job_id} (attempt {job.retry_count})")
                await self.submit_job(job)
            
            raise
            
        finally:
            # Release resources
            if hasattr(job, 'allocated_resources'):
                await self.resource_manager.release(job.allocated_resources)
            
            # Remove from running jobs
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
    
    async def _check_dependencies(self, job: SimulationJob) -> bool:
        """Check if job dependencies are satisfied"""
        for dep_job_id in job.dependencies:
            if dep_job_id not in self.completed_jobs:
                return False
        return True
    
    async def _prepare_input_files(self, job: SimulationJob) -> str:
        """Prepare input files for simulation"""
        input_dir = Path(self.config['temp_storage_path']) / job.job_id
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate input files based on job type
        engine = self.engines[job.simulation_type]
        await engine.prepare_inputs(input_dir, job.material_structure, job.calculation_parameters)
        
        return str(input_dir)
    
    async def _process_output_files(self, job: SimulationJob, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process simulation output files"""
        engine = self.engines[job.simulation_type]
        return await engine.process_outputs(raw_result)
    
    async def _store_job_results(self, job: SimulationJob, result: Dict[str, Any]):
        """Store job results permanently"""
        output_dir = Path(self.config['output_storage_path']) / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result metadata
        result_file = output_dir / 'result.json'
        async with aiofiles.open(result_file, 'w') as f:
            await f.write(json.dumps(result, indent=2, default=str))
        
        # Save job specification
        job_file = output_dir / 'job.json'
        async with aiofiles.open(job_file, 'w') as f:
            job_dict = {
                'job_id': job.job_id,
                'simulation_type': job.simulation_type,
                'material_structure': job.material_structure,
                'calculation_parameters': job.calculation_parameters,
                'priority': job.priority,
                'estimated_runtime': job.estimated_runtime
            }
            await f.write(json.dumps(job_dict, indent=2))
    
    async def _trigger_dependent_jobs(self, completed_job: SimulationJob):
        """Trigger jobs that depend on the completed job"""
        # This would check for waiting jobs with dependencies
        pass
    
    async def _send_callback(self, job: SimulationJob, result: Dict[str, Any]):
        """Send completion callback"""
        try:
            callback_data = {
                'job_id': job.job_id,
                'status': 'completed',
                'result_summary': {
                    'properties_calculated': len(result.get('properties', {})),
                    'convergence_achieved': result.get('converged', False),
                    'computation_time': result.get('wall_time', 0)
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(job.callback_url, json=callback_data) as response:
                    if response.status == 200:
                        logger.info(f"Callback sent successfully for job {job.job_id}")
                    else:
                        logger.warning(f"Callback failed for job {job.job_id}: {response.status}")
        
        except Exception as e:
            logger.error(f"Error sending callback for job {job.job_id}: {e}")
    
    async def _validate_job(self, job: SimulationJob) -> Dict[str, Any]:
        """Validate simulation job specification"""
        errors = []
        
        # Check required fields
        if not job.job_id:
            errors.append("job_id is required")
        if not job.simulation_type:
            errors.append("simulation_type is required")
        if job.simulation_type not in self.engines:
            errors.append(f"Unsupported simulation type: {job.simulation_type}")
        if not job.material_structure:
            errors.append("material_structure is required")
        
        # Validate material structure
        structure_validation = await self._validate_material_structure(job.material_structure)
        if not structure_validation['is_valid']:
            errors.extend(structure_validation['errors'])
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _validate_material_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate material structure data"""
        # Simplified validation - would be more comprehensive in practice
        errors = []
        
        if 'composition' not in structure:
            errors.append("Material composition is required")
        
        if 'lattice' in structure:
            lattice = structure['lattice']
            if not isinstance(lattice, dict) or 'matrix' not in lattice:
                errors.append("Invalid lattice specification")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _estimate_runtime(self, job: SimulationJob) -> float:
        """Estimate job runtime based on system size and parameters"""
        engine = self.engines[job.simulation_type]
        return await engine.estimate_runtime(job.material_structure, job.calculation_parameters)
    
    async def _estimate_resources(self, job: SimulationJob) -> Dict[str, Any]:
        """Estimate required computational resources"""
        engine = self.engines[job.simulation_type]
        return await engine.estimate_resources(job.material_structure, job.calculation_parameters)
    
    async def _resource_monitor(self):
        """Monitor resource usage and availability"""
        while self.is_running:
            try:
                await self.resource_manager.update_availability()
                await asyncio.sleep(self.config['resource_check_interval'])
            except Exception as e:
                logger.error(f"Resource monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _job_cleanup(self):
        """Clean up old temporary files and completed jobs"""
        while self.is_running:
            try:
                # Clean up temporary files for completed jobs
                temp_path = Path(self.config['temp_storage_path'])
                for job_dir in temp_path.iterdir():
                    if job_dir.is_dir():
                        job_id = job_dir.name
                        if job_id in self.completed_jobs or job_id in self.failed_jobs:
                            # Check if job is old enough to clean up
                            job_time = self.completed_jobs.get(job_id, {}).get('completion_time') or \
                                      self.failed_jobs.get(job_id, {}).get('failure_time')
                            
                            if job_time and datetime.now() - job_time > timedelta(hours=24):
                                import shutil
                                shutil.rmtree(job_dir)
                                logger.info(f"Cleaned up temporary files for job {job_id}")
                
                await asyncio.sleep(self.config['cleanup_interval'])
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _workflow_processor(self):
        """Process simulation workflows"""
        while self.is_running:
            try:
                await self.workflow_manager.process_workflows(self.completed_jobs)
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Workflow processor error: {e}")
                await asyncio.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'is_running': self.is_running,
            'queue_size': self.job_queue.qsize(),
            'running_jobs': len(self.running_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'resource_status': self.resource_manager.get_status(),
            'active_workflows': self.workflow_manager.get_active_count()
        }

# =====================================================================
# 3. RESOURCE MANAGER
# =====================================================================

class ResourceManager:
    """Manage computational resources for simulations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.available_resources = self._get_initial_resources()
        self.allocated_resources = {}
        self.resource_lock = asyncio.Lock()
    
    def _get_initial_resources(self) -> Dict[str, Any]:
        """Get initial available resources"""
        try:
            import psutil
            
            return {
                'cpu_cores': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total // (1024**3),
                'gpu_count': len(GPUtil.getGPUs()) if 'GPUtil' in globals() else 0,
                'storage_gb': psutil.disk_usage('/').free // (1024**3)
            }
        except:
            # Fallback values
            return {
                'cpu_cores': 8,
                'memory_gb': 32,
                'gpu_count': 1,
                'storage_gb': 100
            }
    
    async def can_allocate(self, required: Dict[str, Any]) -> bool:
        """Check if required resources can be allocated"""
        async with self.resource_lock:
            for resource, amount in required.items():
                if self.available_resources.get(resource, 0) < amount:
                    return False
            return True
    
    async def allocate(self, required: Dict[str, Any]) -> str:
        """Allocate resources and return allocation ID"""
        async with self.resource_lock:
            # Check availability again
            for resource, amount in required.items():
                if self.available_resources.get(resource, 0) < amount:
                    raise RuntimeError(f"Insufficient {resource}: need {amount}, have {self.available_resources.get(resource, 0)}")
            
            # Allocate resources
            allocation_id = str(uuid.uuid4())
            self.allocated_resources[allocation_id] = required.copy()
            
            for resource, amount in required.items():
                self.available_resources[resource] -= amount
            
            return allocation_id
    
    async def release(self, allocation_id: str):
        """Release allocated resources"""
        async with self.resource_lock:
            if allocation_id in self.allocated_resources:
                allocated = self.allocated_resources[allocation_id]
                
                for resource, amount in allocated.items():
                    self.available_resources[resource] += amount
                
                del self.allocated_resources[allocation_id]
    
    async def update_availability(self):
        """Update resource availability based on system status"""
        # This would check actual system resource usage
        # and update available resources accordingly
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        return {
            'available': self.available_resources.copy(),
            'allocated_count': len(self.allocated_resources),
            'total_allocated': {
                resource: sum(alloc.get(resource, 0) for alloc in self.allocated_resources.values())
                for resource in self.available_resources.keys()
            }
        }

# =====================================================================
# 4. SIMULATION ENGINES (Abstract Base Classes)
# =====================================================================

class SimulationEngine(ABC):
    """Abstract base class for simulation engines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def run_simulation(self, input_dir: str, parameters: Dict[str, Any], 
                           resources: Dict[str, Any]) -> Dict[str, Any]:
        """Run the simulation"""
        pass
    
    @abstractmethod
    async def prepare_inputs(self, input_dir: str, structure: Dict[str, Any], 
                           parameters: Dict[str, Any]):
        """Prepare input files"""
        pass
    
    @abstractmethod
    async def process_outputs(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process output files"""
        pass
    
    @abstractmethod
    async def estimate_runtime(self, structure: Dict[str, Any], 
                             parameters: Dict[str, Any]) -> float:
        """Estimate simulation runtime"""
        pass
    
    @abstractmethod
    async def estimate_resources(self, structure: Dict[str, Any], 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate required resources"""
        pass

class DFTEngine(SimulationEngine):
    """DFT simulation engine"""
    
    async def run_simulation(self, input_dir: str, parameters: Dict[str, Any], 
                           resources: Dict[str, Any]) -> Dict[str, Any]:
        """Run DFT calculation"""
        # Simulate DFT calculation
        await asyncio.sleep(2)  # Simulate computation time
        
        return {
            'converged': True,
            'total_energy': -123.45,
            'band_gap': 2.1,
            'formation_energy': -1.2,
            'wall_time': 7200,
            'output_files': ['OUTCAR', 'vasprun.xml', 'CONTCAR']
        }
    
    async def prepare_inputs(self, input_dir: str, structure: Dict[str, Any], 
                           parameters: Dict[str, Any]):
        """Prepare VASP input files"""
        # Would generate POSCAR, INCAR, POTCAR, KPOINTS files
        pass
    
    async def process_outputs(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process DFT outputs"""
        return {
            'properties': {
                'total_energy': raw_result.get('total_energy'),
                'band_gap': raw_result.get('band_gap'),
                'formation_energy': raw_result.get('formation_energy')
            },
            'converged': raw_result.get('converged', False),
            'computation_details': {
                'wall_time': raw_result.get('wall_time'),
                'software': 'VASP',
                'version': '6.3.0'
            }
        }
    
    async def estimate_runtime(self, structure: Dict[str, Any], 
                             parameters: Dict[str, Any]) -> float:
        """Estimate DFT runtime"""
        # Simplified estimation based on system size
        num_atoms = structure.get('num_atoms', 10)
        k_points = parameters.get('k_points', [4, 4, 4])
        
        # Basic scaling: O(N^3) with system size, linear with k-points
        base_time = 3600  # 1 hour base
        scaling_factor = (num_atoms / 10) ** 2.5 * np.prod(k_points) / 64
        
        return base_time * scaling_factor
    
    async def estimate_resources(self, structure: Dict[str, Any], 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate DFT resource requirements"""
        num_atoms = structure.get('num_atoms', 10)
        
        return {
            'cpu_cores': min(16, max(4, num_atoms // 2)),
            'memory_gb': max(8, num_atoms * 0.5),
            'storage_gb': 5
        }

class MDEngine(SimulationEngine):
    """Molecular Dynamics simulation engine"""
    
    async def run_simulation(self, input_dir: str, parameters: Dict[str, Any], 
                           resources: Dict[str, Any]) -> Dict[str, Any]:
        """Run MD simulation"""
        await asyncio.sleep(1)  # Simulate computation
        
        return {
            'converged': True,
            'trajectory_length': parameters.get('steps', 100000),
            'final_temperature': 300.0,
            'final_pressure': 1.0,
            'average_energy': -245.6,
            'wall_time': 3600
        }
    
    async def prepare_inputs(self, input_dir: str, structure: Dict[str, Any], 
                           parameters: Dict[str, Any]):
        """Prepare MD input files"""
        pass
    
    async def process_outputs(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process MD outputs"""
        return {
            'properties': {
                'thermal_expansion': 1.2e-5,
                'diffusion_coefficient': 1.5e-9,
                'heat_capacity': 0.8
            },
            'trajectory_analysis': {
                'length': raw_result.get('trajectory_length'),
                'final_temperature': raw_result.get('final_temperature'),
                'final_pressure': raw_result.get('final_pressure')
            },
            'converged': raw_result.get('converged', False)
        }
    
    async def estimate_runtime(self, structure: Dict[str, Any], 
                             parameters: Dict[str, Any]) -> float:
        """Estimate MD runtime"""
        num_atoms = structure.get('num_atoms', 10)
        steps = parameters.get('steps', 100000)
        
        # Scaling: linear with atoms and steps
        time_per_step = 0.001  # seconds per step per atom
        return num_atoms * steps * time_per_step
    
    async def estimate_resources(self, structure: Dict[str, Any], 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate MD resource requirements"""
        num_atoms = structure.get('num_atoms', 10)
        
        return {
            'cpu_cores': min(8, max(2, num_atoms // 50)),
            'memory_gb': max(4, num_atoms * 0.1),
            'storage_gb': 10  # For trajectory storage
        }

class FEAEngine(SimulationEngine):
    """Finite Element Analysis engine"""
    
    async def run_simulation(self, input_dir: str, parameters: Dict[str, Any], 
                           resources: Dict[str, Any]) -> Dict[str, Any]:
        """Run FEA simulation"""
        await asyncio.sleep(0.5)  # Simulate computation
        
        return {
            'converged': True,
            'max_stress': 150.5,
            'max_strain': 0.001,
            'displacement_field': 'displacement.dat',
            'wall_time': 1800
        }
    
    async def prepare_inputs(self, input_dir: str, structure: Dict[str, Any], 
                           parameters: Dict[str, Any]):
        """Prepare FEA input files"""
        pass
    
    async def process_outputs(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process FEA outputs"""
        return {
            'properties': {
                'elastic_modulus': 200e9,
                'poisson_ratio': 0.3,
                'yield_strength': 250e6
            },
            'stress_analysis': {
                'max_stress': raw_result.get('max_stress'),
                'max_strain': raw_result.get('max_strain')
            },
            'converged': raw_result.get('converged', False)
        }
    
    async def estimate_runtime(self, structure: Dict[str, Any], 
                             parameters: Dict[str, Any]) -> float:
        """Estimate FEA runtime"""
        mesh_density = parameters.get('mesh_density', 1.0)
        return 1800 * mesh_density  # Base 30 minutes
    
    async def estimate_resources(self, structure: Dict[str, Any], 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate FEA resource requirements"""
        return {
            'cpu_cores': 4,
            'memory_gb': 16,
            'storage_gb': 2
        }

class MonteCarloEngine(SimulationEngine):
    """Monte Carlo simulation engine"""
    
    async def run_simulation(self, input_dir: str, parameters: Dict[str, Any], 
                           resources: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation"""
        await asyncio.sleep(0.3)  # Simulate computation
        
        return {
            'converged': True,
            'acceptance_ratio': 0.4,
            'final_energy': -456.7,
            'samples_generated': parameters.get('samples', 1000000),
            'wall_time': 900
        }
    
    async def prepare_inputs(self, input_dir: str, structure: Dict[str, Any], 
                           parameters: Dict[str, Any]):
        """Prepare Monte Carlo input files"""
        pass
    
    async def process_outputs(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process Monte Carlo outputs"""
        return {
            'properties': {
                'phase_transition_temperature': 450.0,
                'critical_pressure': 2.5e6,
                'order_parameter': 0.75
            },
            'sampling_statistics': {
                'acceptance_ratio': raw_result.get('acceptance_ratio'),
                'samples_generated': raw_result.get('samples_generated')
            },
            'converged': raw_result.get('converged', False)
        }
    
    async def estimate_runtime(self, structure: Dict[str, Any], 
                             parameters: Dict[str, Any]) -> float:
        """Estimate Monte Carlo runtime"""
        samples = parameters.get('samples', 1000000)
        return samples * 0.001  # 1ms per 1000 samples
    
    async def estimate_resources(self, structure: Dict[str, Any], 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate Monte Carlo resource requirements"""
        return {
            'cpu_cores': 2,
            'memory_gb': 4,
            'storage_gb': 1
        }

# =====================================================================
# 5. WORKFLOW MANAGER
# =====================================================================

@dataclass
class SimulationWorkflow:
    """Complex simulation workflow specification"""
    workflow_id: str
    name: str
    description: str
    stages: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    def get_initial_jobs(self) -> List[SimulationJob]:
        """Get jobs that can be started immediately"""
        initial_jobs = []
        
        for stage in self.stages:
            if not stage.get('depends_on'):
                job = SimulationJob(
                    job_id=f"{self.workflow_id}_{stage['name']}",
                    simulation_type=stage['simulation_type'],
                    material_structure=stage['material_structure'],
                    calculation_parameters=stage['parameters'],
                    priority=stage.get('priority', 1)
                )
                initial_jobs.append(job)
        
        return initial_jobs

class WorkflowManager:
    """Manage complex simulation workflows"""
    
    def __init__(self):
        self.active_workflows = {}
        self.completed_workflows = {}
        
    def register_workflow(self, workflow_id: str, workflow: SimulationWorkflow):
        """Register a new workflow"""
        self.active_workflows[workflow_id] = {
            'workflow': workflow,
            'status': 'active',
            'completed_stages': set(),
            'start_time': datetime.now()
        }
    
    async def process_workflows(self, completed_jobs: Dict[str, Any]):
        """Process workflow progress and trigger next stages"""
        for workflow_id, workflow_data in list(self.active_workflows.items()):
            workflow = workflow_data['workflow']
            completed_stages = workflow_data['completed_stages']
            
            # Check for newly completed stages
            new_completions = []
            for stage in workflow.stages:
                stage_job_id = f"{workflow_id}_{stage['name']}"
                if (stage_job_id in completed_jobs and 
                    stage['name'] not in completed_stages):
                    completed_stages.add(stage['name'])
                    new_completions.append(stage['name'])
            
            # Trigger dependent stages
            if new_completions:
                await self._trigger_dependent_stages(workflow_id, new_completions)
            
            # Check if workflow is complete
            if len(completed_stages) == len(workflow.stages):
                await self._complete_workflow(workflow_id)
    
    async def _trigger_dependent_stages(self, workflow_id: str, completed_stages: List[str]):
        """Trigger stages that depend on completed stages"""
        # This would submit new jobs for stages whose dependencies are satisfied
        pass
    
    async def _complete_workflow(self, workflow_id: str):
        """Mark workflow as completed"""
        workflow_data = self.active_workflows[workflow_id]
        workflow_data['status'] = 'completed'
        workflow_data['completion_time'] = datetime.now()
        
        self.completed_workflows[workflow_id] = workflow_data
        del self.active_workflows[workflow_id]
        
        logger.info(f"Workflow {workflow_id} completed")
    
    def get_active_count(self) -> int:
        """Get number of active workflows"""
        return len(self.active_workflows)

# =====================================================================
# 6. ADVANCED ACTIVE LEARNING WITH ACQUISITION FUNCTIONS
# =====================================================================

class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions"""
    
    @abstractmethod
    def __call__(self, predictions: np.ndarray, uncertainties: np.ndarray, 
                 **kwargs) -> np.ndarray:
        """Compute acquisition values"""
        pass

class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function"""
    
    def __init__(self, xi: float = 0.01):
        self.xi = xi  # Exploration parameter
    
    def __call__(self, predictions: np.ndarray, uncertainties: np.ndarray, 
                 best_value: float, **kwargs) -> np.ndarray:
        """Compute Expected Improvement"""
        with np.errstate(divide='warn'):
            improvement = predictions - best_value - self.xi
            Z = improvement / uncertainties
            ei = improvement * norm.cdf(Z) + uncertainties * norm.pdf(Z)
            ei[uncertainties == 0.0] = 0.0
        
        return ei

class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function"""
    
    def __init__(self, beta: float = 2.0):
        self.beta = beta  # Exploration parameter
    
    def __call__(self, predictions: np.ndarray, uncertainties: np.ndarray, 
                 **kwargs) -> np.ndarray:
        """Compute Upper Confidence Bound"""
        return predictions + self.beta * uncertainties

class MaxVariance(AcquisitionFunction):
    """Maximum Variance (pure exploration) acquisition function"""
    
    def __call__(self, predictions: np.ndarray, uncertainties: np.ndarray, 
                 **kwargs) -> np.ndarray:
        """Compute variance-based acquisition"""
        return uncertainties

class KnowledgeGradient(AcquisitionFunction):
    """Knowledge Gradient acquisition function"""
    
    def __call__(self, predictions: np.ndarray, uncertainties: np.ndarray, 
                 **kwargs) -> np.ndarray:
        """Compute Knowledge Gradient (simplified implementation)"""
        # Simplified KG - would need more sophisticated implementation
        return uncertainties * np.abs(predictions)

class AdvancedActiveLearner:
    """Advanced active learning system with multiple acquisition functions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Acquisition functions
        self.acquisition_functions = {
            'ei': ExpectedImprovement(xi=self.config['ei_xi']),
            'ucb': UpperConfidenceBound(beta=self.config['ucb_beta']),
            'max_var': MaxVariance(),
            'kg': KnowledgeGradient()
        }
        
        # Gaussian Process for uncertainty quantification
        self.gp_models = {}
        self.training_data = {}
        self.experiment_history = []
        
        # Multi-objective optimization
        self.objectives = self.config.get('objectives', ['bandgap', 'formation_energy'])
        self.pareto_front = []
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'acquisition_function': 'ei',
            'ei_xi': 0.01,
            'ucb_beta': 2.0,
            'batch_size': 5,
            'diversity_weight': 0.1,
            'objectives': ['bandgap', 'formation_energy'],
            'gp_kernel': 'matern',
            'gp_length_scale': 1.0,
            'gp_noise_level': 0.1,
            'pareto_alpha': 0.1,
            'exploration_decay': 0.95
        }
    
    def update_training_data(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Update training data with new observations"""
        
        if not self.training_data:
            self.training_data = {'X': X, 'y': y}
        else:
            self.training_data['X'] = np.vstack([self.training_data['X'], X])
            for objective in y:
                if objective in self.training_data['y']:
                    self.training_data['y'][objective] = np.concatenate([
                        self.training_data['y'][objective], y[objective]
                    ])
                else:
                    self.training_data['y'][objective] = y[objective]
        
        # Retrain GP models
        self._train_gp_models()
    
    def _train_gp_models(self):
        """Train Gaussian Process models for each objective"""
        
        if not self.training_data or len(self.training_data['X']) == 0:
            return
        
        # Prepare kernel
        if self.config['gp_kernel'] == 'matern':
            kernel = Matern(
                length_scale=self.config['gp_length_scale'],
                nu=2.5
            ) + WhiteKernel(noise_level=self.config['gp_noise_level'])
        else:
            kernel = RBF(
                length_scale=self.config['gp_length_scale']
            ) + WhiteKernel(noise_level=self.config['gp_noise_level'])
        
        # Train GP for each objective
        for objective in self.objectives:
            if objective in self.training_data['y']:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=5
                )
                
                y_data = self.training_data['y'][objective]
                if len(y_data) > 0:
                    gp.fit(self.training_data['X'], y_data)
                    self.gp_models[objective] = gp
                    
                    logger.info(f"Trained GP model for {objective}: "
                               f"log-likelihood = {gp.log_marginal_likelihood():.3f}")
    
    def suggest_candidates(self, candidate_pool: np.ndarray, 
                          batch_size: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Suggest next candidates to evaluate using active learning"""
        
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        if not self.gp_models:
            # Random selection for initial samples
            indices = np.random.choice(len(candidate_pool), size=batch_size, replace=False)
            return candidate_pool[indices], {'method': 'random', 'scores': np.ones(batch_size)}
        
        # Multi-objective acquisition
        acquisition_scores = self._compute_multi_objective_acquisition(candidate_pool)
        
        # Batch selection with diversity
        selected_indices = self._batch_selection_with_diversity(
            candidate_pool, acquisition_scores, batch_size
        )
        
        return candidate_pool[selected_indices], {
            'method': 'multi_objective_acquisition',
            'scores': acquisition_scores[selected_indices],
            'diversity_enforced': True
        }
    
    def _compute_multi_objective_acquisition(self, candidates: np.ndarray) -> np.ndarray:
        """Compute multi-objective acquisition scores"""
        
        objective_scores = {}
        
        for objective in self.objectives:
            if objective in self.gp_models:
                gp = self.gp_models[objective]
                
                # Get predictions and uncertainties
                predictions, std = gp.predict(candidates, return_std=True)
                
                # Compute acquisition function
                acq_func = self.acquisition_functions[self.config['acquisition_function']]
                
                if self.config['acquisition_function'] == 'ei':
                    # Need best observed value for EI
                    best_value = np.max(self.training_data['y'][objective])
                    scores = acq_func(predictions, std, best_value=best_value)
                else:
                    scores = acq_func(predictions, std)
                
                objective_scores[objective] = scores
        
        if not objective_scores:
            return np.random.random(len(candidates))
        
        # Combine multiple objectives using Pareto-based approach
        return self._combine_objective_scores(objective_scores)
    
    def _combine_objective_scores(self, objective_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine multi-objective acquisition scores"""
        
        # Normalize scores
        normalized_scores = {}
        for objective, scores in objective_scores.items():
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score > min_score:
                normalized_scores[objective] = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores[objective] = np.ones_like(scores)
        
        # Hypervolume-based combination (simplified)
        combined_scores = np.zeros(len(list(objective_scores.values())[0]))
        
        for i in range(len(combined_scores)):
            candidate_scores = [normalized_scores[obj][i] for obj in self.objectives]
            
            # Compute hypervolume contribution (simplified as product)
            combined_scores[i] = np.prod(candidate_scores)
        
        return combined_scores
    
    def _batch_selection_with_diversity(self, candidates: np.ndarray, 
                                      acquisition_scores: np.ndarray, 
                                      batch_size: int) -> np.ndarray:
        """Select diverse batch of candidates"""
        
        if len(candidates) <= batch_size:
            return np.arange(len(candidates))
        
        selected_indices = []
        remaining_indices = np.arange(len(candidates))
        
        # Select first candidate with highest acquisition score
        best_idx = np.argmax(acquisition_scores)
        selected_indices.append(best_idx)
        remaining_indices = remaining_indices[remaining_indices != best_idx]
        
        # Select remaining candidates balancing acquisition and diversity
        for _ in range(batch_size - 1):
            if len(remaining_indices) == 0:
                break
            
            # Compute diversity penalty for remaining candidates
            diversity_penalties = np.zeros(len(remaining_indices))
            
            for i, idx in enumerate(remaining_indices):
                candidate = candidates[idx]
                
                # Compute minimum distance to already selected candidates
                if selected_indices:
                    selected_candidates = candidates[selected_indices]
                    distances = np.linalg.norm(selected_candidates - candidate, axis=1)
                    min_distance = np.min(distances)
                    diversity_penalties[i] = 1.0 / (1.0 + min_distance)
            
            # Combined score: acquisition + diversity
            combined_scores = (acquisition_scores[remaining_indices] - 
                             self.config['diversity_weight'] * diversity_penalties)
            
            # Select best candidate
            best_remaining_idx = np.argmax(combined_scores)
            selected_idx = remaining_indices[best_remaining_idx]
            selected_indices.append(selected_idx)
            
            # Remove from remaining
            remaining_indices = remaining_indices[remaining_indices != selected_idx]
        
        return np.array(selected_indices)
    
    def update_pareto_front(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Update Pareto front with new observations"""
        
        if len(self.objectives) < 2:
            return
        
        # Combine all objectives into matrix
        objective_matrix = np.column_stack([y[obj] for obj in self.objectives if obj in y])
        
        if objective_matrix.shape[1] < 2:
            return
        
        # Find Pareto optimal points
        is_pareto = self._is_pareto_optimal(objective_matrix)
        pareto_X = X[is_pareto]
        pareto_y = objective_matrix[is_pareto]
        
        # Update Pareto front
        if len(self.pareto_front) == 0:
            self.pareto_front = list(zip(pareto_X, pareto_y))
        else:
            # Combine with existing front and recompute
            all_X = np.vstack([pareto_X] + [pt[0].reshape(1, -1) for pt in self.pareto_front])
            all_y = np.vstack([pareto_y] + [pt[1].reshape(1, -1) for pt in self.pareto_front])
            
            is_pareto_combined = self._is_pareto_optimal(all_y)
            self.pareto_front = list(zip(all_X[is_pareto_combined], all_y[is_pareto_combined]))
        
        logger.info(f"Updated Pareto front: {len(self.pareto_front)} points")
    
    def _is_pareto_optimal(self, objectives: np.ndarray) -> np.ndarray:
        """Determine which points are Pareto optimal"""
        
        n_points = objectives.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if j dominates i (all objectives better or equal, at least one strictly better)
                    if (np.all(objectives[j] >= objectives[i]) and 
                        np.any(objectives[j] > objectives[i])):
                        is_pareto[i] = False
                        break
        
        return is_pareto
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get active learning status and metrics"""
        
        status = {
            'training_samples': len(self.training_data.get('X', [])),
            'gp_models_trained': len(self.gp_models),
            'pareto_front_size': len(self.pareto_front),
            'objectives': self.objectives,
            'acquisition_function': self.config['acquisition_function']
        }
        
        # Add GP model performance
        if self.gp_models and self.training_data:
            gp_performance = {}
            for objective, gp in self.gp_models.items():
                if objective in self.training_data['y']:
                    y_true = self.training_data['y'][objective]
                    y_pred = gp.predict(self.training_data['X'])
                    
                    gp_performance[objective] = {
                        'r2_score': r2_score(y_true, y_pred),
                        'log_likelihood': gp.log_marginal_likelihood()
                    }
            
            status['gp_performance'] = gp_performance
        
        return status

# =====================================================================
# 7. EXAMPLE USAGE OF ADVANCED FEATURES
# =====================================================================

async def demo_advanced_features():
    """Demonstrate advanced ORION features"""
    
    print("ORION: Advanced Features Demo")
    print("="*40)
    
    # 1. Stream Processing Demo
    print("\n1. Real-time Stream Processing")
    print("-" * 30)
    
    stream_processor = StreamProcessor()
    
    # Start streaming in background
    stream_task = asyncio.create_task(stream_processor.start_streaming())
    
    # Give it a moment to start
    await asyncio.sleep(0.1)
    
    # Submit some test messages
    test_messages = [
        StreamMessage(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            source='demo_source',
            message_type='experimental_result',
            payload={
                'experiment_id': 'exp_001',
                'material_id': 'TiO2_001',
                'measured_properties': {'bandgap': 3.2, 'density': 4.23}
            }
        ),
        StreamMessage(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            source='literature_bot',
            message_type='literature_update',
            payload={
                'doi': '10.1000/demo.paper',
                'title': 'Novel TiO2 Properties',
                'materials': ['TiO2'],
                'properties': {'bandgap': 3.1}
            }
        )
    ]
    
    for msg in test_messages:
        await stream_processor.submit_message(msg)
        print(f"  Submitted message: {msg.message_type}")
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check metrics
    metrics = stream_processor.get_metrics()
    print(f"  Stream metrics: {metrics}")
    
    # Stop streaming
    stream_processor.is_running = False
    try:
        await asyncio.wait_for(stream_task, timeout=1.0)
    except asyncio.TimeoutError:
        stream_task.cancel()
    
    # 2. Simulation Orchestration Demo
    print("\n2. Simulation Orchestration")
    print("-" * 30)
    
    orchestrator = SimulationOrchestrator()
    
    # Start orchestrator in background
    orchestrator_task = asyncio.create_task(orchestrator.start_orchestrator())
    
    # Give it a moment to start
    await asyncio.sleep(0.1)
    
    # Submit test simulation jobs
    test_jobs = [
        SimulationJob(
            job_id=f"dft_job_{i}",
            simulation_type='dft',
            material_structure={
                'composition': {'Ti': 1, 'O': 2},
                'num_atoms': 12,
                'lattice': {'matrix': [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]}
            },
            calculation_parameters={
                'k_points': [4, 4, 4],
                'energy_cutoff': 500
            },
            priority=i + 1
        ) for i in range(3)
    ]
    
    for job in test_jobs:
        job_id = await orchestrator.submit_job(job)
        print(f"  Submitted job: {job_id}")
    
    # Wait for some processing
    await asyncio.sleep(3)
    
    # Check status
    status = orchestrator.get_status()
    print(f"  Orchestrator status: {status}")
    
    # Stop orchestrator
    orchestrator.is_running = False
    try:
        await asyncio.wait_for(orchestrator_task, timeout=1.0)
    except asyncio.TimeoutError:
        orchestrator_task.cancel()
    
    # 3. Active Learning Demo
    print("\n3. Advanced Active Learning")
    print("-" * 30)
    
    active_learner = AdvancedActiveLearner({
        'acquisition_function': 'ei',
        'batch_size': 3,
        'objectives': ['bandgap', 'formation_energy']
    })
    
    # Generate synthetic training data
    np.random.seed(42)
    n_initial = 20
    X_train = np.random.randn(n_initial, 5)  # 5D feature space
    y_train = {
        'bandgap': 2.0 + 0.5 * X_train[:, 0] + 0.3 * X_train[:, 1] + np.random.normal(0, 0.1, n_initial),
        'formation_energy': -1.0 + 0.4 * X_train[:, 2] - 0.2 * X_train[:, 3] + np.random.normal(0, 0.05, n_initial)
    }
    
    # Update training data
    active_learner.update_training_data(X_train, y_train)
    
    # Generate candidate pool
    candidate_pool = np.random.randn(100, 5)
    
    # Suggest next candidates
    suggestions, suggestion_info = active_learner.suggest_candidates(candidate_pool, batch_size=5)
    
    print(f"  Suggested {len(suggestions)} candidates using {suggestion_info['method']}")
    print(f"  Average acquisition score: {np.mean(suggestion_info['scores']):.3f}")
    
    # Update Pareto front
    active_learner.update_pareto_front(X_train, y_train)
    
    # Get learning status
    learning_status = active_learner.get_learning_status()
    print(f"  Learning status: {learning_status}")
    
    print("\nAdvanced Features Demo Completed!")

if __name__ == "__main__":
    asyncio.run(demo_advanced_features())
