"""
Workflow DSL for NANO-OS.

Enables defining complex workflows in YAML or JSON format.

Session 28: Python SDK and Workflow DSL
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml
import json
from pydantic import BaseModel, Field
from uuid import UUID

from .client import NanoOSClient

logger = logging.getLogger(__name__)


class WorkflowStep(BaseModel):
    """Single step in a workflow."""
    name: str = Field(..., description="Step name")
    type: str = Field(..., description="Step type (structure_generation, ml_prediction, etc.)")
    inputs: List[str] = Field(default_factory=list, description="Input variables from previous steps")
    params: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    outputs: List[str] = Field(default_factory=list, description="Output variable names")
    condition: Optional[str] = Field(None, description="Conditional execution (Python expression)")


class WorkflowSpec(BaseModel):
    """Workflow specification."""
    name: str = Field(..., description="Workflow name")
    version: str = Field(default="1.0", description="Workflow version")
    description: Optional[str] = Field(None, description="Workflow description")
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class WorkflowContext:
    """Runtime context for workflow execution."""

    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.step_results: Dict[str, Any] = {}

    def set(self, name: str, value: Any) -> None:
        """Set variable in context."""
        self.variables[name] = value

    def get(self, name: str) -> Any:
        """Get variable from context."""
        return self.variables.get(name)

    def get_all(self) -> Dict[str, Any]:
        """Get all variables."""
        return self.variables.copy()


class WorkflowRunner:
    """
    Execute NANO-OS workflows from YAML/JSON DSL.

    Supports:
    - Structure generation
    - ML predictions
    - DFT calculations
    - Campaign execution
    - Experiment submission
    - Data filtering and transformations

    Example workflow YAML:
        ```yaml
        name: "TMD Screening"
        version: "1.0"
        steps:
          - name: "generate_structures"
            type: "structure_generation"
            params:
              elements: ["Mo", "W", "S", "Se"]
              num_structures: 10
            outputs:
              - structures

          - name: "predict_bandgap"
            type: "ml_prediction"
            inputs:
              - structures
            params:
              model: "cgcnn_bandgap_v1"
            outputs:
              - predictions
        ```
    """

    def __init__(self, client: NanoOSClient):
        """
        Initialize workflow runner.

        Args:
            client: NANO-OS client instance
        """
        self.client = client
        self.context = WorkflowContext()

        # Step type handlers
        self.step_handlers = {
            "structure_generation": self._handle_structure_generation,
            "ml_prediction": self._handle_ml_prediction,
            "dft_batch": self._handle_dft_batch,
            "campaign": self._handle_campaign,
            "filter": self._handle_filter,
            "experiment": self._handle_experiment,
            "transform": self._handle_transform,
        }

    def run_workflow_file(self, workflow_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and execute workflow from file.

        Args:
            workflow_path: Path to YAML or JSON workflow file

        Returns:
            Workflow execution results
        """
        workflow_path = Path(workflow_path)

        # Load workflow
        with open(workflow_path, "r") as f:
            if workflow_path.suffix in [".yaml", ".yml"]:
                workflow_data = yaml.safe_load(f)
            elif workflow_path.suffix == ".json":
                workflow_data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {workflow_path.suffix}")

        workflow_spec = WorkflowSpec(**workflow_data)
        return self.run_workflow(workflow_spec)

    def run_workflow(self, workflow_spec: WorkflowSpec) -> Dict[str, Any]:
        """
        Execute workflow from specification.

        Args:
            workflow_spec: Workflow specification

        Returns:
            Workflow execution results
        """
        logger.info(f"Starting workflow: {workflow_spec.name} (v{workflow_spec.version})")

        # Reset context
        self.context = WorkflowContext()

        # Execute steps sequentially
        for i, step in enumerate(workflow_spec.steps):
            logger.info(f"Executing step {i+1}/{len(workflow_spec.steps)}: {step.name} ({step.type})")

            # Check condition if specified
            if step.condition:
                if not self._evaluate_condition(step.condition):
                    logger.info(f"Skipping step {step.name} (condition not met)")
                    continue

            # Execute step
            try:
                result = self._execute_step(step)
                self.context.step_results[step.name] = result

                # Store outputs in context
                if step.outputs and result:
                    if len(step.outputs) == 1:
                        self.context.set(step.outputs[0], result)
                    elif isinstance(result, dict):
                        for output_name in step.outputs:
                            if output_name in result:
                                self.context.set(output_name, result[output_name])

                logger.info(f"Step {step.name} completed successfully")

            except Exception as e:
                logger.error(f"Step {step.name} failed: {e}")
                raise RuntimeError(f"Workflow failed at step {step.name}: {e}")

        logger.info(f"Workflow {workflow_spec.name} completed successfully")

        return {
            "workflow_name": workflow_spec.name,
            "status": "COMPLETED",
            "results": self.context.get_all(),
            "step_results": self.context.step_results
        }

    def _execute_step(self, step: WorkflowStep) -> Any:
        """Execute a single workflow step."""
        # Get handler for step type
        handler = self.step_handlers.get(step.type)
        if not handler:
            raise ValueError(f"Unknown step type: {step.type}")

        # Prepare inputs
        inputs = {}
        for input_name in step.inputs:
            inputs[input_name] = self.context.get(input_name)

        # Execute handler
        return handler(step, inputs)

    def _handle_structure_generation(self, step: WorkflowStep, inputs: Dict[str, Any]) -> List[Any]:
        """Generate structures."""
        params = step.params
        elements = params.get("elements", [])
        num_structures = params.get("num_structures", 1)
        dimensionality = params.get("dimensionality", 2)

        # Simple random structure generation (stub)
        # In production, this would call a structure generator
        structures = []
        for i in range(num_structures):
            composition = "".join(elements[:2])  # Simple composition
            structure = self.client.structures.create(
                composition=composition,
                dimensionality=dimensionality,
                metadata={"generated_by": "workflow", "step": step.name}
            )
            structures.append(structure)

        logger.info(f"Generated {len(structures)} structures")
        return structures

    def _handle_ml_prediction(self, step: WorkflowStep, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run ML predictions on structures."""
        structures = inputs.get(step.inputs[0], [])
        model_name = step.params.get("model", "cgcnn_bandgap_v1")
        properties = step.params.get("properties", ["bandgap"])

        predictions = []
        for structure in structures:
            job = self.client.jobs.submit_ml_prediction(
                structure_id=structure.id,
                model_name=model_name,
                properties=properties
            )
            # In production, would wait for job completion
            predictions.append({
                "structure_id": structure.id,
                "job_id": job.id,
                "composition": structure.composition
            })

        logger.info(f"Submitted {len(predictions)} ML prediction jobs")
        return predictions

    def _handle_dft_batch(self, step: WorkflowStep, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Submit batch DFT jobs."""
        structures = inputs.get(step.inputs[0], [])
        functional = step.params.get("functional", "PBE")
        kpoints_density = step.params.get("kpoints_density", 0.03)

        jobs = []
        for structure in structures:
            job = self.client.jobs.submit_dft(
                structure_id=structure.id if hasattr(structure, "id") else structure["structure_id"],
                functional=functional,
                kpoints_density=kpoints_density
            )
            jobs.append({
                "structure_id": structure.id if hasattr(structure, "id") else structure["structure_id"],
                "job_id": job.id
            })

        logger.info(f"Submitted {len(jobs)} DFT jobs")
        return jobs

    def _handle_campaign(self, step: WorkflowStep, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create and run a design campaign."""
        config = step.params.get("config", {})
        num_iterations = step.params.get("num_iterations", 1)

        campaign = self.client.campaigns.create(
            name=step.params.get("name", step.name),
            config=config
        )

        # Run iterations
        result = self.client.campaigns.run_iterations(
            campaign_id=campaign.id,
            num_iterations=num_iterations
        )

        logger.info(f"Campaign {campaign.id} completed {num_iterations} iterations")
        return {
            "campaign_id": campaign.id,
            "iterations": num_iterations,
            "result": result
        }

    def _handle_filter(self, step: WorkflowStep, inputs: Dict[str, Any]) -> List[Any]:
        """Filter data based on criteria."""
        data = inputs.get(step.inputs[0], [])
        criteria = step.params.get("criteria", {})

        filtered = []
        for item in data:
            # Check all criteria
            passes = True
            for key, condition in criteria.items():
                value = item.get(key)
                if value is None:
                    passes = False
                    break

                if "min" in condition and value < condition["min"]:
                    passes = False
                    break
                if "max" in condition and value > condition["max"]:
                    passes = False
                    break
                if "equals" in condition and value != condition["equals"]:
                    passes = False
                    break

            if passes:
                filtered.append(item)

        logger.info(f"Filtered {len(data)} items to {len(filtered)} items")
        return filtered

    def _handle_experiment(self, step: WorkflowStep, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Submit experiment to instrument."""
        instrument_id = step.params.get("instrument_id")
        experiment_type = step.params.get("type", "synthesis")
        parameters = step.params.get("parameters", {})
        linked_structure_id = step.params.get("linked_structure_id")

        experiment = self.client.experiments.submit(
            instrument_id=UUID(instrument_id),
            type=experiment_type,
            parameters=parameters,
            linked_structure_id=UUID(linked_structure_id) if linked_structure_id else None
        )

        logger.info(f"Submitted experiment {experiment.id}")
        return {
            "experiment_id": experiment.id,
            "instrument_id": instrument_id,
            "type": experiment_type
        }

    def _handle_transform(self, step: WorkflowStep, inputs: Dict[str, Any]) -> Any:
        """Transform data using Python expression."""
        data = inputs.get(step.inputs[0], [])
        expression = step.params.get("expression", "data")

        # Simple expression evaluation (limited for security)
        # In production, use a proper expression evaluator
        try:
            # Create safe context
            safe_context = {
                "data": data,
                "len": len,
                "sum": sum,
                "max": max,
                "min": min,
                "sorted": sorted,
            }
            result = eval(expression, {"__builtins__": {}}, safe_context)
            return result
        except Exception as e:
            logger.error(f"Transform expression failed: {e}")
            raise

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate conditional expression."""
        try:
            # Create safe context with variables
            safe_context = self.context.get_all()
            safe_context.update({
                "len": len,
                "sum": sum,
                "max": max,
                "min": min,
            })
            return bool(eval(condition, {"__builtins__": {}}, safe_context))
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
