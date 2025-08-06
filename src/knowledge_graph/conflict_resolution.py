"""
ORION Provenance-Weighted Consensus and Conflict Resolution
==========================================================

Handles conflicting property values from different sources using reliability weights.
"""

import numpy as np
import logging
import time
import threading
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from queue import Queue, Empty
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SourceMetadata:
    """Metadata for literature sources"""
    doi: str
    publication_date: str
    citation_count: int
    journal_impact_factor: float
    source_type: str  # 'journal', 'preprint', 'patent', etc.
    confidence_score: float = 1.0  # Manual curation score
    authors: List[str] = field(default_factory=list)
    institution: Optional[str] = None
    experimental_method: Optional[str] = None
    computational_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'doi': self.doi,
            'publication_date': self.publication_date,
            'citation_count': self.citation_count,
            'journal_impact_factor': self.journal_impact_factor,
            'source_type': self.source_type,
            'confidence_score': self.confidence_score,
            'authors': self.authors,
            'institution': self.institution,
            'experimental_method': self.experimental_method,
            'computational_method': self.computational_method
        }


class ProvenanceWeightedConsensus:
    """Handle conflicting property values using source reliability"""
    
    def __init__(self):
        self.source_reliability = {}  # DOI -> reliability score
        self.journal_reliability = {}  # Journal -> base reliability
        self.property_variance_thresholds = {
            'bandgap': 0.2,  # eV
            'formation_energy': 0.1,  # eV/atom
            'bulk_modulus': 10.0,  # GPa
            'density': 0.5,  # g/cm³
            'melting_point': 50.0,  # K
            'thermal_conductivity': 5.0,  # W/m·K
            'electrical_conductivity': 1.0,  # log scale
            'shear_modulus': 5.0,  # GPa
            'youngs_modulus': 10.0,  # GPa
            'poisson_ratio': 0.05,  # dimensionless
            'lattice_constant': 0.1,  # Å
            'volume_per_atom': 1.0,  # Ų
            'cohesive_energy': 0.2,  # eV/atom
            'work_function': 0.1,  # eV
        }
        
        # Method reliability weights
        self.method_weights = {
            'experimental': {
                'single_crystal_xrd': 1.0,
                'powder_xrd': 0.9,
                'neutron_diffraction': 0.95,
                'tem': 0.85,
                'sem': 0.7,
                'afm': 0.75,
                'spectroscopy': 0.8,
                'calorimetry': 0.9,
                'mechanical_testing': 0.85,
                'electrical_measurement': 0.85,
                'unknown': 0.5
            },
            'computational': {
                'dft_hybrid': 0.9,
                'dft_gga': 0.8,
                'dft_lda': 0.7,
                'gw': 0.95,
                'dmft': 0.9,
                'ccsd': 0.95,
                'mp2': 0.85,
                'classical_md': 0.6,
                'ml_prediction': 0.7,
                'unknown': 0.5
            }
        }
        
    def compute_source_weight(self, metadata: SourceMetadata) -> float:
        """Compute reliability weight for a source"""
        
        # Base weight from citation count (logarithmic scaling)
        citation_weight = np.log10(max(metadata.citation_count, 1) + 1)
        
        # Freshness factor (prefer recent publications)
        current_year = datetime.now().year
        try:
            pub_year = int(metadata.publication_date[:4])
        except:
            pub_year = current_year - 5  # Default to 5 years old
            
        age_years = current_year - pub_year
        freshness_factor = np.exp(-age_years / 10.0)  # Half-life of 10 years
        
        # Journal impact factor
        impact_weight = min(metadata.journal_impact_factor / 10.0, 3.0)  # Cap at 3x weight
        
        # Source type modifier
        type_modifiers = {
            'journal': 1.0,
            'preprint': 0.7,
            'patent': 0.8,
            'thesis': 0.6,
            'conference': 0.9,
            'database': 1.1,  # Curated databases get bonus
            'book': 0.85
        }
        type_modifier = type_modifiers.get(metadata.source_type, 0.5)
        
        # Method reliability
        method_weight = 1.0
        if metadata.experimental_method:
            method_weight = self.method_weights['experimental'].get(
                metadata.experimental_method.lower(), 0.5
            )
        elif metadata.computational_method:
            method_weight = self.method_weights['computational'].get(
                metadata.computational_method.lower(), 0.5
            )
        
        # Combine weights
        total_weight = (citation_weight * freshness_factor * impact_weight * 
                       type_modifier * method_weight * metadata.confidence_score)
        
        return max(total_weight, 0.1)  # Minimum weight
    
    def resolve_property_conflict(self, property_values: List[Tuple[float, SourceMetadata]], 
                                property_name: str) -> Dict[str, Any]:
        """Resolve conflicting property values"""
        
        if len(property_values) == 1:
            value, metadata = property_values[0]
            return {
                'consensus_value': value,
                'uncertainty': 0.0,
                'num_sources': 1,
                'weight_sum': self.compute_source_weight(metadata),
                'is_disputed': False,
                'source_spread': 0.0,
                'confidence': 1.0,
                'primary_source': metadata.doi
            }
        
        # Compute weights
        values = []
        weights = []
        sources = []
        
        for value, metadata in property_values:
            weight = self.compute_source_weight(metadata)
            values.append(value)
            weights.append(weight)
            sources.append(metadata)
            
            # Update source reliability tracking
            self.source_reliability[metadata.doi] = weight
        
        values = np.array(values)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Weighted consensus
        consensus_value = np.average(values, weights=weights)
        
        # Uncertainty estimation
        weighted_variance = np.average((values - consensus_value)**2, weights=weights)
        uncertainty = np.sqrt(weighted_variance)
        
        # Detect disputes
        threshold = self.property_variance_thresholds.get(property_name, 0.1)
        is_disputed = uncertainty > threshold
        
        source_spread = np.max(values) - np.min(values)
        
        # Confidence score based on agreement and weights
        max_weight_idx = np.argmax(weights)
        confidence = weights[max_weight_idx]
        
        # Check for outliers
        z_scores = np.abs((values - consensus_value) / (uncertainty + 1e-10))
        outliers = [(i, values[i], sources[i].doi) for i in range(len(values)) if z_scores[i] > 3]
        
        result = {
            'consensus_value': consensus_value,
            'uncertainty': uncertainty,
            'num_sources': len(values),
            'weight_sum': np.sum(weights),
            'is_disputed': is_disputed,
            'source_spread': source_spread,
            'confidence': confidence,
            'primary_source': sources[max_weight_idx].doi,
            'all_values': list(values),
            'all_weights': list(weights),
            'outliers': outliers
        }
        
        if is_disputed:
            logger.warning(f"Disputed {property_name}: consensus={consensus_value:.3f}, "
                         f"spread={source_spread:.3f}, uncertainty={uncertainty:.3f}")
        
        return result
    
    def get_reliability_report(self) -> Dict[str, Any]:
        """Generate report on source reliability"""
        if not self.source_reliability:
            return {'message': 'No sources tracked yet'}
        
        reliabilities = list(self.source_reliability.values())
        
        return {
            'num_sources': len(self.source_reliability),
            'mean_reliability': np.mean(reliabilities),
            'std_reliability': np.std(reliabilities),
            'min_reliability': np.min(reliabilities),
            'max_reliability': np.max(reliabilities),
            'top_sources': sorted(
                self.source_reliability.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }


class ConflictResolutionService:
    """Service for handling property conflicts across the system"""
    
    def __init__(self):
        self.consensus_resolver = ProvenanceWeightedConsensus()
        self.conflict_cache = {}  # Cache resolved conflicts
        self.cache_ttl = 3600  # 1 hour
        self.update_queue = Queue()
        self.is_running = False
        self.processing_thread = None
        self.processing_lock = threading.Lock()
        
    def start(self):
        """Start the conflict resolution service"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._process_updates, 
                daemon=True
            )
            self.processing_thread.start()
            logger.info("Conflict resolution service started")
    
    def stop(self):
        """Stop the conflict resolution service"""
        if self.is_running:
            self.is_running = False
            if self.processing_thread:
                self.processing_thread.join()
            logger.info("Conflict resolution service stopped")
    
    def _process_updates(self):
        """Process conflict resolution updates"""
        while self.is_running:
            try:
                # Get update from queue with timeout
                update = self.update_queue.get(timeout=1.0)
                
                if update['type'] == 'resolve':
                    self._handle_resolution(update)
                elif update['type'] == 'update_reliability':
                    self._update_source_reliability(update)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing update: {e}")
    
    def _handle_resolution(self, update: Dict[str, Any]):
        """Handle a conflict resolution request"""
        material_id = update['material_id']
        property_name = update['property_name']
        property_values = update['property_values']
        
        # Resolve conflict
        resolution = self.consensus_resolver.resolve_property_conflict(
            property_values, property_name
        )
        
        # Cache result
        cache_key = f"{material_id}:{property_name}"
        self.conflict_cache[cache_key] = {
            'resolution': resolution,
            'timestamp': time.time()
        }
        
        # Log if disputed
        if resolution['is_disputed']:
            logger.info(f"Resolved disputed {property_name} for {material_id}: "
                       f"{resolution['consensus_value']:.3f} ± {resolution['uncertainty']:.3f}")
    
    def _update_source_reliability(self, update: Dict[str, Any]):
        """Update source reliability based on validation results"""
        doi = update['doi']
        validation_result = update['validation_result']
        
        # Adjust reliability based on validation
        current_reliability = self.consensus_resolver.source_reliability.get(doi, 1.0)
        
        if validation_result['is_valid']:
            # Increase reliability
            new_reliability = min(current_reliability * 1.1, 5.0)
        else:
            # Decrease reliability
            new_reliability = max(current_reliability * 0.9, 0.1)
        
        self.consensus_resolver.source_reliability[doi] = new_reliability
        logger.debug(f"Updated reliability for {doi}: {current_reliability:.2f} -> {new_reliability:.2f}")
    
    def resolve_property(self, material_id: str, property_name: str, 
                        property_values: List[Tuple[float, SourceMetadata]]) -> Dict[str, Any]:
        """Resolve conflicting property values for a material"""
        
        # Check cache
        cache_key = f"{material_id}:{property_name}"
        if cache_key in self.conflict_cache:
            cached = self.conflict_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['resolution']
        
        # Queue for processing
        update = {
            'type': 'resolve',
            'material_id': material_id,
            'property_name': property_name,
            'property_values': property_values
        }
        self.update_queue.put(update)
        
        # For synchronous operation, process immediately
        resolution = self.consensus_resolver.resolve_property_conflict(
            property_values, property_name
        )
        
        # Update cache
        self.conflict_cache[cache_key] = {
            'resolution': resolution,
            'timestamp': time.time()
        }
        
        return resolution
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and statistics"""
        with self.processing_lock:
            queue_size = self.update_queue.qsize()
        
        return {
            'is_running': self.is_running,
            'queue_size': queue_size,
            'cache_size': len(self.conflict_cache),
            'total_sources': len(self.consensus_resolver.source_reliability),
            'reliability_stats': self.consensus_resolver.get_reliability_report()
        }
    
    def clear_cache(self):
        """Clear the conflict resolution cache"""
        self.conflict_cache.clear()
        logger.info("Conflict resolution cache cleared")
    
    def export_reliability_scores(self) -> Dict[str, float]:
        """Export current source reliability scores"""
        return dict(self.consensus_resolver.source_reliability)
    
    def import_reliability_scores(self, scores: Dict[str, float]):
        """Import source reliability scores"""
        self.consensus_resolver.source_reliability.update(scores)
        logger.info(f"Imported reliability scores for {len(scores)} sources")