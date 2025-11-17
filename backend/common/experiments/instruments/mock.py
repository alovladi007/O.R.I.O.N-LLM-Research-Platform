"""
Mock instrument adapter for testing and demonstration.

Simulates lab equipment behavior without requiring actual hardware.

Session 21: Lab Integration & Experiment Management
"""

import logging
import time
import random
from typing import Dict, Any, Optional
import numpy as np

from .base import InstrumentAdapter, ExperimentExecutionResult

logger = logging.getLogger(__name__)


class MockInstrumentAdapter(InstrumentAdapter):
    """
    Mock instrument adapter for testing.

    Simulates various types of lab equipment:
    - Synthesis: CVD, sputtering, etc.
    - Measurement: XRD, SEM, optical spectroscopy
    - Characterization: TEM, XPS, Raman

    Returns realistic but fake results.

    Example:
        >>> adapter = MockInstrumentAdapter({
        ...     "instrument_type": "CVD",
        ...     "simulation_delay": 2.0
        ... })
        >>> with adapter:
        ...     result = adapter.execute_experiment(
        ...         experiment_type="synthesis",
        ...         parameters={"temperature": 800, "duration": 3600}
        ...     )
        >>> print(result.results)
    """

    def __init__(self, connection_info: Dict[str, Any]):
        """
        Initialize mock adapter.

        Args:
            connection_info: Configuration including:
                - instrument_type: Type of instrument to simulate
                - simulation_delay: Delay in seconds to simulate processing
                - error_rate: Probability of random errors (0-1)
        """
        super().__init__(connection_info)
        self.instrument_type = connection_info.get("instrument_type", "generic")
        self.simulation_delay = connection_info.get("simulation_delay", 1.0)
        self.error_rate = connection_info.get("error_rate", 0.0)
        self.experiment_count = 0

    def connect(self) -> bool:
        """
        Simulate connection to instrument.

        Returns:
            True (always succeeds for mock)
        """
        logger.info(f"Connecting to mock {self.instrument_type} instrument...")
        time.sleep(0.1)  # Simulate connection delay
        self.is_connected = True
        logger.info("Connected successfully")
        return True

    def disconnect(self) -> None:
        """Simulate disconnection."""
        if self.is_connected:
            logger.info("Disconnecting from mock instrument...")
            self.is_connected = False

    def execute_experiment(
        self,
        experiment_type: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExperimentExecutionResult:
        """
        Execute mock experiment.

        Args:
            experiment_type: Type of experiment
            parameters: Experiment parameters
            metadata: Optional metadata

        Returns:
            ExperimentExecutionResult with simulated results
        """
        if not self.is_connected:
            return ExperimentExecutionResult(
                success=False,
                results={},
                error_message="Instrument not connected"
            )

        logger.info(
            f"Executing {experiment_type} experiment with parameters: {parameters}"
        )

        # Simulate random errors
        if random.random() < self.error_rate:
            logger.error("Random simulated error occurred")
            return ExperimentExecutionResult(
                success=False,
                results={},
                error_message="Simulated random error during execution"
            )

        # Simulate processing time
        time.sleep(self.simulation_delay)

        # Generate fake results based on experiment type
        results = self._generate_results(experiment_type, parameters)

        self.experiment_count += 1
        external_job_id = f"mock_{self.instrument_type}_{self.experiment_count}"

        logger.info(f"Experiment completed successfully. Job ID: {external_job_id}")

        return ExperimentExecutionResult(
            success=True,
            results=results,
            external_job_id=external_job_id,
            metadata={
                "simulation": True,
                "instrument_type": self.instrument_type,
                "execution_time": self.simulation_delay
            }
        )

    def check_status(self) -> Dict[str, Any]:
        """
        Check mock instrument status.

        Returns:
            Status information
        """
        return {
            "online": self.is_connected,
            "busy": False,  # Mock is never busy
            "instrument_type": self.instrument_type,
            "experiments_completed": self.experiment_count,
            "health": "good"
        }

    def _generate_results(
        self,
        experiment_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate fake but realistic results.

        Args:
            experiment_type: Type of experiment
            parameters: Experiment parameters

        Returns:
            Simulated results
        """
        if experiment_type == "synthesis":
            return self._generate_synthesis_results(parameters)
        elif experiment_type == "measurement":
            return self._generate_measurement_results(parameters)
        elif experiment_type == "characterization":
            return self._generate_characterization_results(parameters)
        elif experiment_type == "testing":
            return self._generate_testing_results(parameters)
        else:
            return self._generate_generic_results(parameters)

    def _generate_synthesis_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fake synthesis results."""
        temperature = params.get("temperature", 500)
        duration = params.get("duration", 3600)

        # Temperature affects yield and quality
        optimal_temp = 800
        temp_factor = 1.0 - abs(temperature - optimal_temp) / optimal_temp

        yield_percent = max(0, min(100, 70 + 20 * temp_factor + random.gauss(0, 5)))
        thickness_nm = max(0, duration / 10 + random.gauss(0, 10))
        uniformity = max(0, min(100, 80 + 10 * temp_factor + random.gauss(0, 5)))

        return {
            "yield_percent": round(yield_percent, 2),
            "thickness_nm": round(thickness_nm, 2),
            "uniformity_percent": round(uniformity, 2),
            "deposition_rate_nm_min": round(thickness_nm / (duration / 60), 3),
            "quality": "good" if yield_percent > 80 else "acceptable" if yield_percent > 60 else "poor"
        }

    def _generate_measurement_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fake measurement results."""
        measurement_type = params.get("measurement_type", "optical")

        if measurement_type == "optical":
            # Fake optical measurements
            wavelengths = np.linspace(400, 800, 50)  # nm
            # Fake absorption spectrum (Gaussian peak)
            peak_pos = random.uniform(500, 700)
            absorption = np.exp(-((wavelengths - peak_pos) ** 2) / (2 * 50 ** 2))
            absorption += random.normal(0, 0.05, len(wavelengths))

            # Estimate bandgap from absorption edge
            bandgap_ev = 1240 / peak_pos  # Simple conversion

            return {
                "wavelengths_nm": wavelengths.tolist(),
                "absorption": absorption.tolist(),
                "peak_wavelength_nm": round(peak_pos, 2),
                "estimated_bandgap_eV": round(bandgap_ev, 3),
                "measurement_type": "UV-Vis spectroscopy"
            }

        elif measurement_type == "electrical":
            # Fake electrical measurements
            voltages = np.linspace(-2, 2, 41)
            # Fake I-V curve (diode-like)
            currents = np.exp(voltages * 5) - 1
            currents += random.normal(0, 0.1, len(currents))

            return {
                "voltages_V": voltages.tolist(),
                "currents_mA": currents.tolist(),
                "resistance_ohm": round(1000 * random.uniform(0.8, 1.2), 2),
                "measurement_type": "I-V curve"
            }

        else:
            # Generic measurement
            return {
                "measured_value": round(random.uniform(1, 100), 2),
                "unit": "arbitrary",
                "measurement_type": measurement_type
            }

    def _generate_characterization_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fake characterization results."""
        method = params.get("method", "XRD")

        if method == "XRD":
            # Fake XRD pattern
            two_theta = np.linspace(10, 80, 100)
            # Fake peaks at typical positions
            intensity = np.zeros_like(two_theta)
            peak_positions = [14.4, 29.0, 39.5, 49.8]  # Typical MoS2 peaks
            for pos in peak_positions:
                intensity += 1000 * np.exp(-((two_theta - pos) ** 2) / (2 * 0.5 ** 2))
            intensity += random.normal(0, 50, len(two_theta))

            return {
                "two_theta": two_theta.tolist(),
                "intensity": intensity.tolist(),
                "peak_positions": peak_positions,
                "phase_identified": "hexagonal",
                "lattice_constant_a": round(3.16 + random.gauss(0, 0.02), 3),
                "method": "XRD"
            }

        elif method == "SEM":
            return {
                "image_file": "mock_sem_image.tif",
                "magnification": params.get("magnification", 10000),
                "grain_size_nm": round(random.uniform(50, 500), 1),
                "surface_roughness_nm": round(random.uniform(1, 20), 2),
                "method": "SEM"
            }

        else:
            return {
                "method": method,
                "result": "characterization completed",
                "data_file": f"mock_{method}_data.dat"
            }

    def _generate_testing_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fake testing results."""
        test_type = params.get("test_type", "cycling")

        if test_type == "cycling":
            # Fake battery cycling data
            cycles = range(1, 101)
            capacity = [100 * (0.98 ** i) + random.gauss(0, 1) for i in cycles]

            return {
                "cycles": list(cycles),
                "capacity_mAh_g": capacity,
                "capacity_retention_percent": round(capacity[-1] / capacity[0] * 100, 2),
                "test_type": "battery cycling"
            }

        else:
            return {
                "test_type": test_type,
                "result": "pass",
                "metric_value": round(random.uniform(0, 100), 2)
            }

    def _generate_generic_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic fake results."""
        return {
            "status": "completed",
            "generic_metric_1": round(random.uniform(0, 100), 2),
            "generic_metric_2": round(random.uniform(0, 100), 2),
            "notes": "Mock experiment completed successfully"
        }
