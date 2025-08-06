"""
ORION Candidate Generator
========================

Main interface for generating novel material candidates.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import uuid
import numpy as np
from .advanced_generator import EnsembleSurrogate, DiversitySampler
from ..core.physics_validator import PhysicsSanityChecker

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """
    Advanced candidate generator with ensemble models and physics validation.
    
    This implements:
    - LLM-guided material generation
    - Structure prediction with uncertainty
    - Property targeting with constraints
    - Novelty scoring and diversity
    - Synthesizability assessment
    - Physics validation
    """
    
    def __init__(self, config, knowledge_graph=None, rag_system=None):
        self.config = config
        self.knowledge_graph = knowledge_graph
        self.rag_system = rag_system
        self._initialized = False
        
        # Advanced components
        self.ensemble_surrogate = EnsembleSurrogate()
        self.diversity_sampler = DiversitySampler(
            diversity_weight=config.get('diversity_weight', 0.3),
            risk_aversion=config.get('risk_aversion', 1.0)
        )
        self.physics_checker = PhysicsSanityChecker()
        
        # Inject physics checker into ensemble
        self.ensemble_surrogate.physics_checker = self.physics_checker
        
        logger.info("Advanced Candidate Generator created")
    
    async def initialize(self):
        """Initialize candidate generator"""
        # Load pre-trained surrogate models if available
        model_dir = self.config.get('surrogate_model_dir')
        if model_dir:
            try:
                from .surrogate_trainer import SurrogatePredictorTrainer
                trainer = SurrogatePredictorTrainer(self.config)
                self.ensemble_surrogate = trainer.load_models(model_dir)
                logger.info(f"Loaded surrogate models from {model_dir}")
            except Exception as e:
                logger.warning(f"Could not load surrogate models: {e}")
        
        self._initialized = True
        logger.info("Candidate Generator initialized")
    
    async def shutdown(self):
        """Shutdown candidate generator"""
        self._initialized = False
        logger.info("Candidate Generator shutdown")
    
    async def generate_candidates(self, query: str, constraints: Dict[str, Any], 
                                 num_candidates: int = 5) -> List[Dict[str, Any]]:
        """Generate material candidates with advanced features"""
        
        # Extract target properties and constraints
        target_properties = constraints.get('target_properties', {})
        composition_constraints = constraints.get('composition', {})
        structure_constraints = constraints.get('structure', {})
        
        # Use RAG to find similar materials
        similar_materials = []
        if self.rag_system:
            rag_results = await self.rag_system.search(
                query, 
                num_results=20,
                filters=constraints
            )
            similar_materials = rag_results.get('materials', [])
        
        # Generate candidate compositions
        candidates = []
        
        # Strategy 1: Modify existing materials
        for material in similar_materials[:10]:
            variations = self._generate_variations(material, target_properties)
            candidates.extend(variations)
        
        # Strategy 2: Generate novel compositions
        if self.knowledge_graph:
            novel_compositions = await self._generate_novel_compositions(
                target_properties, 
                composition_constraints,
                num_candidates=num_candidates * 2
            )
            candidates.extend(novel_compositions)
        
        # Predict properties with uncertainty
        if self.ensemble_surrogate and candidates:
            predictions, uncertainties = self.ensemble_surrogate.predict(
                candidates, 
                return_uncertainty=True
            )
            
            # Add predictions to candidates
            for i, candidate in enumerate(candidates):
                candidate['predictions'] = {
                    prop: predictions[prop][i] if prop in predictions else None
                    for prop in target_properties
                }
                candidate['uncertainties'] = {
                    prop: uncertainties[prop][i] if prop in uncertainties else 0.0
                    for prop in target_properties
                }
        
        # Validate physics
        for candidate in candidates:
            is_valid, errors = self.physics_checker.validate_candidate(candidate)
            candidate['physics_valid'] = is_valid
            candidate['physics_errors'] = errors
            
            # Compute stability score
            if 'predictions' in candidate:
                candidate['stability_score'] = self.physics_checker.compute_stability_score(
                    candidate['predictions']
                )
        
        # Rank and select diverse candidates
        ranked_candidates = self.diversity_sampler.rank_candidates(
            candidates,
            target_property=list(target_properties.keys())[0] if target_properties else 'formation_energy',
            target_value=list(target_properties.values())[0] if target_properties else 0.0,
            return_details=True
        )
        
        # Select top diverse candidates
        selected_indices = self.diversity_sampler.select_batch(
            candidates,
            batch_size=num_candidates,
            target_property=list(target_properties.keys())[0] if target_properties else 'formation_energy',
            target_value=list(target_properties.values())[0] if target_properties else 0.0
        )
        
        final_candidates = []
        for idx in selected_indices:
            candidate = candidates[idx]
            candidate['id'] = str(uuid.uuid4())
            candidate['generation_method'] = 'advanced_ensemble'
            final_candidates.append(candidate)
        
        return final_candidates
    
    def _generate_variations(self, material: Dict[str, Any], 
                           target_properties: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate variations of existing material"""
        variations = []
        
        composition = material.get('composition', {})
        if not composition:
            return variations
        
        # Substitution strategies
        substitution_rules = {
            'Ti': ['Zr', 'Hf', 'V'],
            'O': ['S', 'Se', 'N'],
            'Si': ['Ge', 'Sn', 'C'],
            'Fe': ['Co', 'Ni', 'Mn'],
            'Cu': ['Ag', 'Au', 'Zn']
        }
        
        # Generate substitutions
        for element, substitutes in substitution_rules.items():
            if element in composition:
                for substitute in substitutes:
                    new_comp = composition.copy()
                    new_comp[substitute] = new_comp.pop(element)
                    
                    variations.append({
                        'composition': new_comp,
                        'parent_material': material.get('material_id', 'unknown'),
                        'modification': f'{element}->{substitute}',
                        'structure': material.get('structure', {})
                    })
        
        # Doping strategies (add small amounts)
        dopants = ['B', 'N', 'P', 'F']
        for dopant in dopants:
            if dopant not in composition:
                new_comp = composition.copy()
                # Reduce one element slightly and add dopant
                main_element = max(composition.items(), key=lambda x: x[1])[0]
                if composition[main_element] > 0.1:
                    new_comp[main_element] -= 0.05
                    new_comp[dopant] = 0.05
                    
                    variations.append({
                        'composition': new_comp,
                        'parent_material': material.get('material_id', 'unknown'),
                        'modification': f'{dopant}-doped',
                        'structure': material.get('structure', {})
                    })
        
        return variations[:5]  # Limit variations
    
    async def _generate_novel_compositions(self, target_properties: Dict[str, float],
                                         composition_constraints: Dict[str, Any],
                                         num_candidates: int = 10) -> List[Dict[str, Any]]:
        """Generate completely novel compositions"""
        candidates = []
        
        # Get element statistics from knowledge graph
        if self.knowledge_graph:
            common_elements = await self.knowledge_graph.get_common_elements_for_property(
                list(target_properties.keys())[0] if target_properties else 'bandgap'
            )
        else:
            # Fallback to common elements
            common_elements = ['O', 'Si', 'Al', 'Ti', 'Fe', 'C', 'N', 'S', 'P', 'B']
        
        # Generate random compositions
        for _ in range(num_candidates):
            n_elements = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            selected_elements = np.random.choice(common_elements, size=n_elements, replace=False)
            
            # Generate stoichiometry
            fractions = np.random.dirichlet(np.ones(n_elements))
            
            composition = {}
            for elem, frac in zip(selected_elements, fractions):
                composition[elem] = float(frac)
            
            # Apply constraints
            if composition_constraints:
                if 'forbidden_elements' in composition_constraints:
                    forbidden = set(composition_constraints['forbidden_elements'])
                    if any(elem in forbidden for elem in composition):
                        continue
                
                if 'required_elements' in composition_constraints:
                    required = set(composition_constraints['required_elements'])
                    if not all(elem in composition for elem in required):
                        continue
            
            candidates.append({
                'composition': composition,
                'generation_method': 'novel_random',
                'structure': self._predict_structure(composition)
            })
        
        return candidates
    
    def _predict_structure(self, composition: Dict[str, float]) -> Dict[str, Any]:
        """Predict crystal structure for composition"""
        # Simplified structure prediction
        # In real implementation, this would use ML models or heuristics
        
        # Count different element types
        metals = ['Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
        nonmetals = ['O', 'S', 'Se', 'N', 'P', 'C', 'Si', 'F', 'Cl', 'Br']
        
        n_metals = sum(1 for elem in composition if elem in metals)
        n_nonmetals = sum(1 for elem in composition if elem in nonmetals)
        
        # Simple heuristics
        if n_metals == 0:
            # Nonmetal only - molecular or covalent
            crystal_system = 'monoclinic'
        elif n_nonmetals == 0:
            # Metal only - likely metallic
            crystal_system = 'cubic'
        elif n_metals == 1 and n_nonmetals == 1:
            # Binary compound
            if 'O' in composition:
                crystal_system = 'cubic'  # Many oxides are cubic
            else:
                crystal_system = 'hexagonal'
        else:
            # Complex compound
            crystal_system = np.random.choice(['orthorhombic', 'tetragonal', 'monoclinic'])
        
        return {
            'crystal_system': crystal_system,
            'space_group': 'P1',  # Placeholder
            'lattice_parameters': {
                'a': np.random.uniform(3, 10),
                'b': np.random.uniform(3, 10),
                'c': np.random.uniform(3, 10),
                'alpha': 90.0,
                'beta': 90.0,
                'gamma': 90.0
            }
        }
    
    async def rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank candidates by multiple criteria"""
        # Use diversity sampler for ranking
        target_property = 'formation_energy'  # Default
        if candidates and 'predictions' in candidates[0]:
            # Use first predicted property
            target_property = list(candidates[0]['predictions'].keys())[0]
        
        rankings = self.diversity_sampler.rank_candidates(
            candidates,
            target_property=target_property,
            maximize=False,
            return_details=True
        )
        
        # Sort candidates based on rankings
        ranked_candidates = []
        for idx, score, details in rankings:
            candidate = candidates[idx].copy()
            candidate['ranking_score'] = score
            candidate['ranking_details'] = details
            ranked_candidates.append(candidate)
        
        return ranked_candidates