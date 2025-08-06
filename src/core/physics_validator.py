"""
ORION Physics Sanity Check Layer
================================

Validates predictions against physical constraints and thermodynamic consistency.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class PhysicsSanityChecker:
    """Validate predictions against physical constraints"""
    
    def __init__(self):
        self.constraints = {
            'bandgap': {'min': 0.0, 'max': 15.0, 'unit': 'eV'},
            'density': {'min': 0.1, 'max': 25.0, 'unit': 'g/cm³'},
            'melting_point': {'min': 0.0, 'max': 4000.0, 'unit': 'K'},
            'dielectric_constant': {'min': 1.0, 'max': 10000.0, 'unit': 'dimensionless'},
            'bulk_modulus': {'min': 0.1, 'max': 500.0, 'unit': 'GPa'},
            'shear_modulus': {'min': 0.1, 'max': 300.0, 'unit': 'GPa'},
            'youngs_modulus': {'min': 0.1, 'max': 1000.0, 'unit': 'GPa'},
            'thermal_conductivity': {'min': 0.001, 'max': 2000.0, 'unit': 'W/m·K'},
            'electrical_conductivity': {'min': 1e-20, 'max': 1e8, 'unit': 'S/m'},
            'formation_energy': {'min': -10.0, 'max': 10.0, 'unit': 'eV/atom'},
            'poisson_ratio': {'min': -1.0, 'max': 0.5, 'unit': 'dimensionless'},
            'heat_capacity': {'min': 0.0, 'max': 1000.0, 'unit': 'J/mol·K'},
            'thermal_expansion': {'min': -1e-4, 'max': 1e-3, 'unit': '1/K'},
            'refractive_index': {'min': 1.0, 'max': 5.0, 'unit': 'dimensionless'},
            'work_function': {'min': 1.0, 'max': 10.0, 'unit': 'eV'},
            'electron_affinity': {'min': -3.0, 'max': 5.0, 'unit': 'eV'},
            'ionization_energy': {'min': 3.0, 'max': 25.0, 'unit': 'eV'},
            'lattice_constant': {'min': 2.0, 'max': 20.0, 'unit': 'Å'},
            'volume_per_atom': {'min': 5.0, 'max': 100.0, 'unit': 'Ų'},
            'cohesive_energy': {'min': 0.1, 'max': 10.0, 'unit': 'eV/atom'}
        }
        
        # Chemical composition constraints
        self.element_constraints = {
            'electronegativity_difference': {'max': 4.0},
            'atomic_radius_ratio': {'min': 0.2, 'max': 5.0},
            'valence_electron_range': {'min': 1, 'max': 8},
            'oxidation_state_range': {'min': -4, 'max': 8}
        }
        
        # Relationships between properties
        self.property_relationships = {
            'mechanical': {
                'bulk_shear_ratio': {'min': 1.0, 'max': 3.0},  # Bulk/Shear modulus
                'poisson_from_moduli': {'min': -1.0, 'max': 0.5}
            },
            'electronic': {
                'bandgap_conductivity': 'inverse',  # High bandgap -> low conductivity
                'work_function_range': {'min': 0.5, 'max': 2.0}  # Times bandgap
            },
            'thermal': {
                'debye_temperature_melting': {'ratio': 0.3}  # Debye temp ~ 0.3 * melting point
            }
        }
    
    def check_property_constraints(self, predictions: Dict[str, float]) -> Dict[str, bool]:
        """Check if predicted properties satisfy physical constraints"""
        results = {}
        violations = []
        
        for prop, value in predictions.items():
            if prop in self.constraints:
                constraint = self.constraints[prop]
                is_valid = constraint['min'] <= value <= constraint['max']
                results[prop] = is_valid
                
                if not is_valid:
                    violations.append(
                        f"{prop}: {value:.3f} {constraint['unit']} "
                        f"(valid range: {constraint['min']}-{constraint['max']} {constraint['unit']})"
                    )
        
        if violations:
            logger.warning(f"Physics constraint violations: {'; '.join(violations)}")
        
        return results
    
    def check_thermodynamic_consistency(self, predictions: Dict[str, float]) -> bool:
        """Check thermodynamic relationships between properties"""
        checks = []
        
        # Check if formation energy is reasonable for stability
        if 'formation_energy' in predictions:
            fe = predictions['formation_energy']
            if fe > 2.0:  # Very positive formation energy suggests instability
                checks.append(False)
                logger.warning(f"High formation energy ({fe:.3f} eV/atom) suggests thermodynamic instability")
            else:
                checks.append(True)
        
        # Check mechanical property relationships (Bulk modulus should be > Shear modulus)
        if 'bulk_modulus' in predictions and 'shear_modulus' in predictions:
            bulk = predictions['bulk_modulus']
            shear = predictions['shear_modulus']
            is_valid = bulk >= shear
            checks.append(is_valid)
            if not is_valid:
                logger.warning(f"Bulk modulus ({bulk:.1f}) < Shear modulus ({shear:.1f})")
        
        # Check Poisson's ratio bounds if calculable
        if all(k in predictions for k in ['bulk_modulus', 'shear_modulus']):
            bulk = predictions['bulk_modulus']
            shear = predictions['shear_modulus']
            if shear > 0:
                poisson = (3*bulk - 2*shear) / (6*bulk + 2*shear)
                is_valid = -1.0 <= poisson <= 0.5
                checks.append(is_valid)
                if not is_valid:
                    logger.warning(f"Calculated Poisson's ratio ({poisson:.3f}) outside physical bounds [-1, 0.5]")
        
        # Check Young's modulus consistency
        if all(k in predictions for k in ['bulk_modulus', 'shear_modulus', 'youngs_modulus']):
            bulk = predictions['bulk_modulus']
            shear = predictions['shear_modulus']
            youngs = predictions['youngs_modulus']
            if bulk > 0 and shear > 0:
                expected_youngs = 9 * bulk * shear / (3 * bulk + shear)
                deviation = abs(youngs - expected_youngs) / expected_youngs
                is_valid = deviation < 0.2  # 20% tolerance
                checks.append(is_valid)
                if not is_valid:
                    logger.warning(f"Young's modulus ({youngs:.1f}) inconsistent with bulk/shear moduli (expected: {expected_youngs:.1f})")
        
        # Check electronic property relationships
        if 'bandgap' in predictions and 'electrical_conductivity' in predictions:
            bg = predictions['bandgap']
            cond = predictions['electrical_conductivity']
            # Semiconductors/insulators should have low conductivity
            if bg > 0.5 and cond > 1e5:
                checks.append(False)
                logger.warning(f"High electrical conductivity ({cond:.1e} S/m) inconsistent with bandgap ({bg:.2f} eV)")
            else:
                checks.append(True)
        
        # Check thermal properties
        if 'melting_point' in predictions and 'thermal_conductivity' in predictions:
            mp = predictions['melting_point']
            tc = predictions['thermal_conductivity']
            # Very high melting point materials typically have high thermal conductivity
            if mp > 2500 and tc < 1.0:
                logger.info(f"Low thermal conductivity ({tc:.2f} W/m·K) for high melting point ({mp:.0f} K) material")
        
        # Check cohesive energy vs formation energy
        if 'cohesive_energy' in predictions and 'formation_energy' in predictions:
            cohesive = predictions['cohesive_energy']
            formation = predictions['formation_energy']
            # Cohesive energy should be positive and typically larger than |formation energy|
            if cohesive < 0:
                checks.append(False)
                logger.warning(f"Negative cohesive energy ({cohesive:.3f} eV/atom)")
            elif cohesive < abs(formation) * 0.5:
                logger.info(f"Low cohesive energy ({cohesive:.3f}) compared to formation energy ({formation:.3f})")
        
        return all(checks) if checks else True
    
    def validate_candidate(self, candidate: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Comprehensive validation of a material candidate"""
        errors = []
        
        # Extract predictions
        if 'predictions' not in candidate:
            errors.append("No predictions found in candidate")
            return False, errors
        
        predictions = candidate['predictions']
        
        # Check property constraints
        constraint_results = self.check_property_constraints(predictions)
        failed_constraints = [prop for prop, valid in constraint_results.items() if not valid]
        if failed_constraints:
            errors.extend([f"Property constraint violation: {prop}" for prop in failed_constraints])
        
        # Check thermodynamic consistency
        if not self.check_thermodynamic_consistency(predictions):
            errors.append("Thermodynamic inconsistency detected")
        
        # Check chemical composition if available
        if 'composition' in candidate:
            comp_errors = self._validate_composition(candidate['composition'])
            errors.extend(comp_errors)
        
        # Check crystal structure if available
        if 'structure' in candidate:
            struct_errors = self._validate_structure(candidate['structure'])
            errors.extend(struct_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_composition(self, composition: Dict[str, float]) -> List[str]:
        """Validate chemical composition"""
        errors = []
        
        # Check if composition sums to ~1.0
        total = sum(composition.values())
        if not (0.95 <= total <= 1.05):
            errors.append(f"Composition sum ({total:.3f}) not close to 1.0")
        
        # Check for reasonable stoichiometry
        values = list(composition.values())
        if any(v <= 0 for v in values):
            errors.append("Negative or zero composition fractions")
        
        if max(values) > 0.9:  # Single element dominance
            errors.append("Single element dominance (>90%) may not form compound")
        
        # Check charge balance (simplified)
        if len(composition) > 1:
            # This would require oxidation state information
            pass
        
        return errors
    
    def _validate_structure(self, structure: Dict[str, Any]) -> List[str]:
        """Validate crystal structure parameters"""
        errors = []
        
        if 'lattice_parameters' in structure:
            params = structure['lattice_parameters']
            
            # Check lattice parameter ranges
            for param in ['a', 'b', 'c']:
                if param in params:
                    value = params[param]
                    if not (2.0 <= value <= 20.0):
                        errors.append(f"Lattice parameter {param}={value:.2f} Å outside reasonable range")
            
            # Check angles for different crystal systems
            if 'angles' in params and 'crystal_system' in structure:
                angles = params['angles']
                system = structure['crystal_system']
                
                if system == 'cubic' and not all(abs(a - 90) < 0.1 for a in angles.values()):
                    errors.append("Cubic system requires all angles = 90°")
                elif system == 'hexagonal':
                    if abs(angles.get('alpha', 90) - 90) > 0.1 or abs(angles.get('beta', 90) - 90) > 0.1:
                        errors.append("Hexagonal system requires α = β = 90°")
                    if abs(angles.get('gamma', 120) - 120) > 0.1:
                        errors.append("Hexagonal system requires γ = 120°")
        
        return errors
    
    def suggest_corrections(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Suggest corrections for invalid predictions"""
        corrections = {}
        
        # Apply constraints
        for prop, value in predictions.items():
            if prop in self.constraints:
                constraint = self.constraints[prop]
                if value < constraint['min']:
                    corrections[prop] = constraint['min']
                elif value > constraint['max']:
                    corrections[prop] = constraint['max']
                else:
                    corrections[prop] = value
            else:
                corrections[prop] = value
        
        # Ensure mechanical property consistency
        if all(k in corrections for k in ['bulk_modulus', 'shear_modulus']):
            if corrections['shear_modulus'] > corrections['bulk_modulus']:
                # Adjust shear modulus to be slightly less than bulk
                corrections['shear_modulus'] = corrections['bulk_modulus'] * 0.9
        
        return corrections
    
    def compute_stability_score(self, predictions: Dict[str, float]) -> float:
        """Compute overall stability score based on multiple factors"""
        score = 1.0
        
        # Formation energy contribution
        if 'formation_energy' in predictions:
            fe = predictions['formation_energy']
            if fe < -3.0:
                score *= 1.0  # Very stable
            elif fe < 0:
                score *= 0.9  # Stable
            elif fe < 0.5:
                score *= 0.7  # Metastable
            else:
                score *= 0.3  # Unstable
        
        # Cohesive energy contribution
        if 'cohesive_energy' in predictions:
            ce = predictions['cohesive_energy']
            if ce > 5.0:
                score *= 1.0  # Strong bonding
            elif ce > 2.0:
                score *= 0.8
            else:
                score *= 0.5  # Weak bonding
        
        # Mechanical stability
        if all(k in predictions for k in ['bulk_modulus', 'shear_modulus']):
            bulk = predictions['bulk_modulus']
            shear = predictions['shear_modulus']
            
            # Born stability criteria
            if bulk > 0 and shear > 0 and bulk > shear:
                score *= 1.0
            else:
                score *= 0.2  # Mechanically unstable
        
        return max(0.0, min(1.0, score))