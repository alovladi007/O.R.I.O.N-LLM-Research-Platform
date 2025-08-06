#!/usr/bin/env python3
"""
ORION Quick Start Example
========================

This script demonstrates the basic capabilities of the ORION platform.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import ORIONSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main example function"""
    
    # Initialize ORION system
    logger.info("Initializing ORION system...")
    orion = ORIONSystem()
    
    try:
        await orion.initialize()
        logger.info("ORION system initialized successfully!")
        
        # Example 1: Search for materials with specific properties
        logger.info("\n" + "="*50)
        logger.info("Example 1: Searching for semiconductors with bandgap 1.5-2.0 eV")
        logger.info("="*50)
        
        response = await orion.process_query(
            "Find semiconductor materials with bandgap between 1.5 and 2.0 eV"
        )
        
        if response["status"] == "success":
            results = response["result"]["results"]
            logger.info(f"Found {len(results)} materials:")
            for i, material in enumerate(results[:5], 1):
                logger.info(f"{i}. {material.get('formula', 'Unknown')} - "
                          f"Bandgap: {material.get('bandgap', 'N/A')} eV")
        
        # Example 2: Generate new material candidates
        logger.info("\n" + "="*50)
        logger.info("Example 2: Generating new photovoltaic materials")
        logger.info("="*50)
        
        response = await orion.process_query(
            "Design new materials for solar cells with high efficiency and stability"
        )
        
        if response["status"] == "success":
            candidates = response["result"]["candidates"]
            logger.info(f"Generated {len(candidates)} candidates:")
            for i, candidate in enumerate(candidates[:3], 1):
                logger.info(f"\n{i}. {candidate.get('formula', 'Unknown')}")
                logger.info(f"   Score: {candidate.get('score', 0):.3f}")
                logger.info(f"   Predicted efficiency: {candidate.get('efficiency', 'N/A')}%")
                logger.info(f"   Stability score: {candidate.get('stability', 'N/A')}")
        
        # Example 3: Design an experimental protocol
        logger.info("\n" + "="*50)
        logger.info("Example 3: Generating synthesis protocol")
        logger.info("="*50)
        
        response = await orion.process_query(
            "Create a protocol for synthesizing TiO2 nanoparticles using sol-gel method"
        )
        
        if response["status"] == "success":
            protocol = response["result"]["protocol"]
            logger.info(f"Protocol: {protocol.get('title', 'TiO2 Synthesis')}")
            logger.info(f"Method: {protocol.get('method', 'Sol-gel')}")
            logger.info(f"Duration: {protocol.get('duration', 'N/A')}")
            logger.info(f"Temperature: {protocol.get('temperature', 'N/A')}°C")
            logger.info(f"Safety notes: {len(protocol.get('safety_notes', []))} items")
        
        # Example 4: Submit a simulation
        logger.info("\n" + "="*50)
        logger.info("Example 4: Submitting DFT calculation")
        logger.info("="*50)
        
        response = await orion.process_query(
            "Calculate the band structure of GaN using DFT"
        )
        
        if response["status"] == "success":
            job_info = response["result"]
            logger.info(f"Simulation job submitted:")
            logger.info(f"Job ID: {job_info.get('job_id', 'N/A')}")
            logger.info(f"Status: {job_info.get('status', 'N/A')}")
            logger.info(f"Estimated time: {job_info.get('estimated_time', 'N/A')}")
        
        # Example 5: Get system status
        logger.info("\n" + "="*50)
        logger.info("Example 5: System Status")
        logger.info("="*50)
        
        status = await orion.get_system_status()
        
        logger.info("System Information:")
        logger.info(f"- Version: {status['system_info']['version']}")
        logger.info(f"- Status: {status['status']}")
        logger.info(f"- Modules initialized: {sum(status['modules'].values())}/{len(status['modules'])}")
        
        if 'knowledge_graph_stats' in status:
            kg_stats = status['knowledge_graph_stats']
            logger.info("\nKnowledge Graph Statistics:")
            logger.info(f"- Materials: {kg_stats.get('materials', 0)}")
            logger.info(f"- Properties: {kg_stats.get('properties', 0)}")
            logger.info(f"- Processes: {kg_stats.get('processes', 0)}")
            logger.info(f"- Total relationships: {kg_stats.get('relationships', 0)}")
        
        performance = status.get('performance', {})
        if performance:
            logger.info("\nPerformance Metrics:")
            logger.info(f"- CPU Usage: {performance.get('cpu_usage', {}).get('mean', 0):.1f}%")
            logger.info(f"- Memory Usage: {performance.get('memory_usage', {}).get('mean', 0):.1f}%")
        
        # Example 6: Ingest data (if file exists)
        data_file = Path("examples/sample_materials.json")
        if data_file.exists():
            logger.info("\n" + "="*50)
            logger.info("Example 6: Data Ingestion")
            logger.info("="*50)
            
            result = await orion.ingest_data(
                str(data_file),
                data_type="materials_project"
            )
            
            logger.info(f"Ingestion completed:")
            logger.info(f"- Total processed: {result.get('total_processed', 0)}")
            logger.info(f"- Nodes created: {result.get('nodes_created', 0)}")
            logger.info(f"- Relationships created: {result.get('relationships_created', 0)}")
            logger.info(f"- Errors: {result.get('errors', 0)}")
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        
    finally:
        # Shutdown ORION
        logger.info("\nShutting down ORION system...")
        await orion.shutdown()
        logger.info("ORION shutdown complete.")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())