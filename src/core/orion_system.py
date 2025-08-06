"""
ORION System Core Orchestrator
=============================

Main system orchestrator that coordinates all ORION modules.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json

from .config_manager import ConfigManager, get_config
from .monitoring import PerformanceMonitor, BottleneckAnalyzer
from .exceptions import ORIONException, ConfigurationError

# Import module interfaces (to be implemented)
from ..knowledge_graph import KnowledgeGraphManager
from ..rag import RAGSystem
from ..candidate_generation import CandidateGenerator
from ..simulation import SimulationOrchestrator
from ..experimental_design import ExperimentalDesigner
from ..feedback_loop import FeedbackLoop
from ..data_ingest import DataIngestPipeline

logger = logging.getLogger(__name__)


class ORIONSystem:
    """
    Main ORION system orchestrator.
    
    Coordinates all modules and provides a unified interface for:
    - Literature mining and knowledge graph management
    - Candidate material generation
    - Simulation orchestration
    - Experimental design
    - Feedback and active learning
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize ORION system.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing ORION system...")
        
        # Configuration
        self.config = ConfigManager(config_path) if config_path else get_config()
        if not self.config.validate():
            raise ConfigurationError("Invalid configuration")
        
        # System metadata
        self.system_info = {
            "name": self.config.get("app.name", "ORION"),
            "version": self.config.get("app.version", "1.0.0"),
            "tagline": self.config.get("app.tagline", "Charting new frontiers in material science"),
            "initialized_at": datetime.now().isoformat(),
        }
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        self.bottleneck_analyzer = BottleneckAnalyzer(self.monitor)
        
        # Core modules (lazy initialization)
        self._knowledge_graph: Optional[KnowledgeGraphManager] = None
        self._rag_system: Optional[RAGSystem] = None
        self._candidate_generator: Optional[CandidateGenerator] = None
        self._simulation_orchestrator: Optional[SimulationOrchestrator] = None
        self._experimental_designer: Optional[ExperimentalDesigner] = None
        self._feedback_loop: Optional[FeedbackLoop] = None
        self._data_ingest: Optional[DataIngestPipeline] = None
        
        # System state
        self._initialized = False
        self._running = False
        
        logger.info(f"ORION system created: {self.system_info['name']} v{self.system_info['version']}")
    
    async def initialize(self):
        """Initialize all system modules"""
        if self._initialized:
            logger.warning("System already initialized")
            return
        
        logger.info("Initializing ORION modules...")
        
        try:
            # Start monitoring
            self.monitor.start()
            self.bottleneck_analyzer.start()
            
            # Initialize modules in dependency order
            await self._initialize_knowledge_graph()
            await self._initialize_rag_system()
            await self._initialize_candidate_generator()
            await self._initialize_simulation_orchestrator()
            await self._initialize_experimental_designer()
            await self._initialize_feedback_loop()
            await self._initialize_data_ingest()
            
            self._initialized = True
            self._running = True
            
            logger.info("ORION system initialization complete")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            await self.shutdown()
            raise ORIONException(f"Failed to initialize ORION system: {e}")
    
    async def _initialize_knowledge_graph(self):
        """Initialize knowledge graph module"""
        logger.info("Initializing knowledge graph...")
        try:
            self._knowledge_graph = KnowledgeGraphManager(self.config.knowledge_graph)
            await self._knowledge_graph.initialize()
        except Exception as e:
            logger.error(f"Knowledge graph initialization failed: {e}")
            raise
    
    async def _initialize_rag_system(self):
        """Initialize RAG system"""
        logger.info("Initializing RAG system...")
        try:
            self._rag_system = RAGSystem(
                config=self.config.rag,
                knowledge_graph=self._knowledge_graph
            )
            await self._rag_system.initialize()
        except Exception as e:
            logger.error(f"RAG system initialization failed: {e}")
            raise
    
    async def _initialize_candidate_generator(self):
        """Initialize candidate generator"""
        logger.info("Initializing candidate generator...")
        try:
            self._candidate_generator = CandidateGenerator(
                config=self.config,
                knowledge_graph=self._knowledge_graph,
                rag_system=self._rag_system
            )
            await self._candidate_generator.initialize()
        except Exception as e:
            logger.error(f"Candidate generator initialization failed: {e}")
            raise
    
    async def _initialize_simulation_orchestrator(self):
        """Initialize simulation orchestrator"""
        logger.info("Initializing simulation orchestrator...")
        try:
            self._simulation_orchestrator = SimulationOrchestrator(
                config=self.config.get_section("simulation")
            )
            await self._simulation_orchestrator.initialize()
        except Exception as e:
            logger.error(f"Simulation orchestrator initialization failed: {e}")
            raise
    
    async def _initialize_experimental_designer(self):
        """Initialize experimental designer"""
        logger.info("Initializing experimental designer...")
        try:
            self._experimental_designer = ExperimentalDesigner(
                config=self.config.get_section("experimental"),
                knowledge_graph=self._knowledge_graph
            )
            await self._experimental_designer.initialize()
        except Exception as e:
            logger.error(f"Experimental designer initialization failed: {e}")
            raise
    
    async def _initialize_feedback_loop(self):
        """Initialize feedback loop"""
        logger.info("Initializing feedback loop...")
        try:
            self._feedback_loop = FeedbackLoop(
                config=self.config,
                knowledge_graph=self._knowledge_graph,
                rag_system=self._rag_system
            )
            await self._feedback_loop.initialize()
        except Exception as e:
            logger.error(f"Feedback loop initialization failed: {e}")
            raise
    
    async def _initialize_data_ingest(self):
        """Initialize data ingestion pipeline"""
        logger.info("Initializing data ingestion pipeline...")
        try:
            self._data_ingest = DataIngestPipeline(
                config=self.config,
                knowledge_graph=self._knowledge_graph
            )
            await self._data_ingest.initialize()
        except Exception as e:
            logger.error(f"Data ingestion pipeline initialization failed: {e}")
            raise
    
    @property
    def knowledge_graph(self) -> KnowledgeGraphManager:
        """Get knowledge graph manager"""
        if not self._knowledge_graph:
            raise ORIONException("Knowledge graph not initialized")
        return self._knowledge_graph
    
    @property
    def rag_system(self) -> RAGSystem:
        """Get RAG system"""
        if not self._rag_system:
            raise ORIONException("RAG system not initialized")
        return self._rag_system
    
    @property
    def candidate_generator(self) -> CandidateGenerator:
        """Get candidate generator"""
        if not self._candidate_generator:
            raise ORIONException("Candidate generator not initialized")
        return self._candidate_generator
    
    @property
    def simulation_orchestrator(self) -> SimulationOrchestrator:
        """Get simulation orchestrator"""
        if not self._simulation_orchestrator:
            raise ORIONException("Simulation orchestrator not initialized")
        return self._simulation_orchestrator
    
    @property
    def experimental_designer(self) -> ExperimentalDesigner:
        """Get experimental designer"""
        if not self._experimental_designer:
            raise ORIONException("Experimental designer not initialized")
        return self._experimental_designer
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a natural language query through the ORION system.
        
        Args:
            query: Natural language query
            context: Optional context information
            
        Returns:
            Response dictionary with results
        """
        if not self._initialized:
            raise ORIONException("System not initialized")
        
        start_time = datetime.now()
        self.monitor.increment_counter("queries")
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Extract intent and entities
            intent_analysis = await self._analyze_query_intent(query, context)
            
            # Route to appropriate handler
            if intent_analysis["intent"] == "generate_material":
                result = await self._handle_material_generation(query, intent_analysis, context)
            elif intent_analysis["intent"] == "simulate":
                result = await self._handle_simulation_request(query, intent_analysis, context)
            elif intent_analysis["intent"] == "design_experiment":
                result = await self._handle_experimental_design(query, intent_analysis, context)
            elif intent_analysis["intent"] == "search":
                result = await self._handle_search_query(query, intent_analysis, context)
            else:
                result = await self._handle_general_query(query, intent_analysis, context)
            
            # Record metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.monitor.record_timing("query_processing", duration)
            
            return {
                "status": "success",
                "query": query,
                "intent": intent_analysis["intent"],
                "result": result,
                "processing_time": duration,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            self.monitor.increment_counter("query_errors")
            raise ORIONException(f"Failed to process query: {e}")
    
    async def _analyze_query_intent(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze query intent and extract entities"""
        # Use RAG system to understand query intent
        rag_response = await self.rag_system.analyze_query(query, context)
        
        # Determine primary intent
        intent_keywords = {
            "generate_material": ["design", "create", "generate", "synthesize", "new material"],
            "simulate": ["simulate", "calculate", "compute", "dft", "molecular dynamics"],
            "design_experiment": ["experiment", "protocol", "procedure", "synthesis method"],
            "search": ["find", "search", "lookup", "what is", "properties of"],
        }
        
        intent = "general"
        for intent_type, keywords in intent_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                intent = intent_type
                break
        
        return {
            "intent": intent,
            "entities": rag_response.get("entities", {}),
            "context": rag_response.get("context", {})
        }
    
    async def _handle_material_generation(self, query: str, intent_analysis: Dict[str, Any], 
                                         context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle material generation requests"""
        logger.info("Handling material generation request")
        
        # Generate candidates
        candidates = await self.candidate_generator.generate_candidates(
            query=query,
            constraints=intent_analysis.get("entities", {}).get("constraints", {}),
            num_candidates=intent_analysis.get("entities", {}).get("num_candidates", 5)
        )
        
        # Score and rank candidates
        ranked_candidates = await self.candidate_generator.rank_candidates(candidates)
        
        # Optionally run initial simulations
        if context and context.get("run_simulations", False):
            for candidate in ranked_candidates[:3]:  # Top 3 candidates
                sim_result = await self.simulation_orchestrator.quick_screen(candidate)
                candidate["simulation_preview"] = sim_result
        
        return {
            "candidates": ranked_candidates,
            "generation_method": "llm_guided",
            "total_generated": len(candidates)
        }
    
    async def _handle_simulation_request(self, query: str, intent_analysis: Dict[str, Any],
                                        context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle simulation requests"""
        logger.info("Handling simulation request")
        
        material = intent_analysis.get("entities", {}).get("material")
        if not material:
            raise ORIONException("No material specified for simulation")
        
        # Determine simulation type
        sim_type = intent_analysis.get("entities", {}).get("simulation_type", "dft")
        
        # Submit simulation job
        job_id = await self.simulation_orchestrator.submit_job(
            material=material,
            simulation_type=sim_type,
            parameters=intent_analysis.get("entities", {}).get("parameters", {})
        )
        
        return {
            "job_id": job_id,
            "status": "submitted",
            "simulation_type": sim_type,
            "estimated_time": "2-4 hours"
        }
    
    async def _handle_experimental_design(self, query: str, intent_analysis: Dict[str, Any],
                                         context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle experimental design requests"""
        logger.info("Handling experimental design request")
        
        material = intent_analysis.get("entities", {}).get("material")
        synthesis_method = intent_analysis.get("entities", {}).get("method")
        
        # Generate experimental protocol
        protocol = await self.experimental_designer.design_protocol(
            material=material,
            method=synthesis_method,
            constraints=intent_analysis.get("entities", {}).get("constraints", {})
        )
        
        return {
            "protocol": protocol,
            "safety_notes": protocol.get("safety_notes", []),
            "equipment_required": protocol.get("equipment", [])
        }
    
    async def _handle_search_query(self, query: str, intent_analysis: Dict[str, Any],
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle search queries"""
        logger.info("Handling search query")
        
        # Search knowledge graph
        kg_results = await self.knowledge_graph.search(
            query=query,
            filters=intent_analysis.get("entities", {}).get("filters", {})
        )
        
        # Enhance with RAG
        enhanced_results = await self.rag_system.enhance_search_results(
            query=query,
            kg_results=kg_results
        )
        
        return {
            "results": enhanced_results,
            "total_found": len(kg_results),
            "sources": [r.get("source") for r in enhanced_results if r.get("source")]
        }
    
    async def _handle_general_query(self, query: str, intent_analysis: Dict[str, Any],
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle general queries"""
        logger.info("Handling general query")
        
        # Use RAG system for general response
        response = await self.rag_system.generate_response(
            query=query,
            context=context
        )
        
        return {
            "response": response["text"],
            "references": response.get("references", []),
            "confidence": response.get("confidence", 0.0)
        }
    
    async def ingest_data(self, data_source: str, data_type: str = "auto") -> Dict[str, Any]:
        """
        Ingest data from various sources.
        
        Args:
            data_source: Path or URL to data source
            data_type: Type of data (auto, paper, patent, experimental)
            
        Returns:
            Ingestion results
        """
        if not self._initialized:
            raise ORIONException("System not initialized")
        
        logger.info(f"Ingesting data from {data_source}")
        
        result = await self._data_ingest.ingest(
            source=data_source,
            data_type=data_type
        )
        
        # Update knowledge graph
        if result["success"]:
            await self.knowledge_graph.update_from_ingestion(result["data"])
            
            # Update RAG indices
            await self.rag_system.update_indices(result["data"])
        
        return result
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "system_info": self.system_info,
            "status": "running" if self._running else "stopped",
            "initialized": self._initialized,
            "modules": {
                "knowledge_graph": self._knowledge_graph is not None,
                "rag_system": self._rag_system is not None,
                "candidate_generator": self._candidate_generator is not None,
                "simulation_orchestrator": self._simulation_orchestrator is not None,
                "experimental_designer": self._experimental_designer is not None,
                "feedback_loop": self._feedback_loop is not None,
                "data_ingest": self._data_ingest is not None,
            },
            "performance": self.monitor.get_metrics_summary(),
            "bottlenecks": self.bottleneck_analyzer.get_bottleneck_report()
        }
        
        # Add module-specific status
        if self._knowledge_graph:
            status["knowledge_graph_stats"] = await self._knowledge_graph.get_stats()
        
        if self._simulation_orchestrator:
            status["simulation_queue"] = await self._simulation_orchestrator.get_queue_status()
        
        return status
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down ORION system...")
        
        self._running = False
        
        # Shutdown modules in reverse order
        shutdown_tasks = []
        
        if self._data_ingest:
            shutdown_tasks.append(self._data_ingest.shutdown())
        if self._feedback_loop:
            shutdown_tasks.append(self._feedback_loop.shutdown())
        if self._experimental_designer:
            shutdown_tasks.append(self._experimental_designer.shutdown())
        if self._simulation_orchestrator:
            shutdown_tasks.append(self._simulation_orchestrator.shutdown())
        if self._candidate_generator:
            shutdown_tasks.append(self._candidate_generator.shutdown())
        if self._rag_system:
            shutdown_tasks.append(self._rag_system.shutdown())
        if self._knowledge_graph:
            shutdown_tasks.append(self._knowledge_graph.shutdown())
        
        # Wait for all shutdowns
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Stop monitoring
        self.bottleneck_analyzer.stop()
        self.monitor.stop()
        
        self._initialized = False
        logger.info("ORION system shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        asyncio.run(self.initialize())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        asyncio.run(self.shutdown())
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()