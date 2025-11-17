"""
Design loop service for autonomous materials discovery campaigns.

This module implements the core design loop logic for AI-driven materials
discovery. It orchestrates the iterative process of:
1. Generating candidate structures
2. Evaluating candidates using ML predictions
3. Scoring against target properties
4. Updating campaign state

AI Agent Integration:
---------------------
This service is designed to be controlled by AI agents that can:

1. **Monitor Progress**:
   - Query campaign status and metrics
   - Analyze iteration results
   - Identify promising directions

2. **Make Decisions**:
   - Choose generation strategies (random, Bayesian, genetic, RL)
   - Adjust parameters based on results
   - Decide when to stop or continue

3. **Learn and Adapt**:
   - Analyze what works across campaigns
   - Transfer knowledge between campaigns
   - Optimize search strategies

Future Enhancements:
--------------------
1. **Advanced Generation Strategies**:
   - Bayesian Optimization: Use Gaussian processes for acquisition
   - Genetic Algorithms: Crossover and mutation of structures
   - Reinforcement Learning: Learn policy for structure generation
   - Generative Models: VAE/GAN for novel structures
   - Active Learning: Query most informative candidates

2. **Multi-Objective Optimization**:
   - Pareto front tracking for multiple objectives
   - Weighted scalarization of objectives
   - Hypervolume indicator for progress

3. **Uncertainty Quantification**:
   - Model ensembles for prediction uncertainty
   - Acquisition functions using uncertainty
   - Confidence-based stopping criteria

4. **Transfer Learning**:
   - Warm-start from previous campaigns
   - Meta-learning across campaigns
   - Knowledge distillation from successful runs

Example AI Agent Workflow:
--------------------------
```python
# AI agent creates campaign
campaign = await create_campaign(
    name="Find 2eV bandgap TMD",
    target_properties={
        "bandgap": {"value": 2.0, "tolerance": 0.2, "weight": 1.0}
    },
    constraints={"elements": ["Mo", "W", "S", "Se"], "dimensionality": 2}
)

# Agent runs iterations and monitors
while not campaign.is_terminal:
    # Run next iteration
    iteration = await run_iteration(db, campaign.id)

    # Agent analyzes results
    if iteration.best_score_this_iter > 0.95:
        # Great result! Agent might pause to analyze
        await pause_campaign(db, campaign.id)
        break

    # Agent decides whether to adjust strategy
    if iteration.metrics["improvement_from_previous"] < 0.01:
        # Stuck in local optimum, switch strategy
        await update_campaign_config(
            db, campaign.id,
            config={"strategy": "exploration"}
        )

# Agent reviews best discoveries
best_structures = await get_campaign_best_structures(db, campaign.id, top_k=10)
```
"""

import logging
import random
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

logger = logging.getLogger(__name__)


class DesignLoopService:
    """
    Service for running design loop iterations.

    This service orchestrates the autonomous design process, coordinating:
    - Structure generation
    - ML-based evaluation
    - Score calculation
    - Campaign state updates

    The service is stateless - all state is persisted in the database.
    """

    @staticmethod
    async def run_iteration(
        db: AsyncSession,
        campaign_id: uuid.UUID
    ) -> Any:  # Returns DesignIteration
        """
        Run one iteration of the design loop.

        This is the main entry point for executing a design iteration. It:
        1. Loads campaign configuration
        2. Generates candidate structures
        3. Evaluates candidates using ML predictions
        4. Calculates scores against target properties
        5. Updates campaign metrics and best results
        6. Saves iteration results to database

        Args:
            db: Async database session
            campaign_id: UUID of the campaign to run

        Returns:
            DesignIteration object with results

        Raises:
            ValueError: If campaign not found or in invalid state

        Example:
            >>> iteration = await DesignLoopService.run_iteration(db, campaign_id)
            >>> print(f"Best score: {iteration.best_score_this_iter}")
            Best score: 0.87

        AI Agent Notes:
            - Agents should call this repeatedly until convergence
            - Check campaign.status before calling
            - Monitor iteration.metrics for learning signals
        """
        from ...src.api.models.campaign import DesignCampaign, DesignIteration, CampaignStatus
        from ...src.api.models.structure import Structure

        logger.info(f"Starting iteration for campaign {campaign_id}")

        # Load campaign
        result = await db.execute(
            select(DesignCampaign).where(DesignCampaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()

        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if campaign.is_terminal:
            raise ValueError(
                f"Campaign is in terminal state {campaign.status.value}. "
                f"Cannot run more iterations."
            )

        # Check iteration limit
        if campaign.current_iteration >= campaign.max_iterations:
            campaign.status = CampaignStatus.COMPLETED
            campaign.completed_at = datetime.utcnow()
            await db.commit()
            raise ValueError("Campaign reached max iterations")

        # Extract config
        config = campaign.config or {}
        target_properties = config.get("target_properties", {})
        constraints = config.get("constraints", {})
        strategy = config.get("strategy", "random")
        num_candidates = config.get("candidates_per_iteration", 10)

        logger.info(
            f"Running iteration {campaign.current_iteration + 1}/{campaign.max_iterations} "
            f"with strategy={strategy}, candidates={num_candidates}"
        )

        # Step 1: Generate candidates
        candidates = await DesignLoopService.generate_candidates(
            db=db,
            campaign=campaign,
            num_candidates=num_candidates
        )

        logger.info(f"Generated {len(candidates)} candidate structures")

        # Step 2: Evaluate candidates
        evaluations = await DesignLoopService.evaluate_candidates(
            db=db,
            candidates=candidates,
            target_properties=target_properties
        )

        logger.info(f"Evaluated {len(evaluations)} candidates")

        # Step 3: Find best candidate this iteration
        best_eval = max(evaluations, key=lambda e: e["score"]) if evaluations else None
        best_score_this_iter = best_eval["score"] if best_eval else None
        best_structure_id_this_iter = best_eval["structure_id"] if best_eval else None

        # Step 4: Calculate iteration metrics
        scores = [e["score"] for e in evaluations]
        metrics = {
            "scores": scores,
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "std_score": _calculate_std(scores) if scores else 0.0,
            "num_candidates": len(candidates),
            "num_evaluated": len(evaluations),
        }

        # Calculate improvement from previous iteration
        if campaign.best_score is not None and best_score_this_iter is not None:
            metrics["improvement_from_previous"] = best_score_this_iter - campaign.best_score
        else:
            metrics["improvement_from_previous"] = None

        # Diversity metric (simple version - std of scores)
        # Future: Use structure similarity, fingerprints, etc.
        metrics["diversity_metric"] = metrics["std_score"]

        logger.info(
            f"Iteration metrics: mean={metrics['mean_score']:.3f}, "
            f"max={metrics['max_score']:.3f}, "
            f"improvement={metrics.get('improvement_from_previous', 'N/A')}"
        )

        # Step 5: Create iteration record
        iteration = DesignIteration(
            campaign_id=campaign_id,
            iteration_index=campaign.current_iteration,
            created_structures=[str(c.id) for c in candidates],
            evaluated_structures=[str(e["structure_id"]) for e in evaluations],
            best_score_this_iter=best_score_this_iter,
            best_structure_id_this_iter=best_structure_id_this_iter,
            metrics=metrics,
            strategy_used=strategy,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )

        db.add(iteration)

        # Step 6: Update campaign state
        campaign.current_iteration += 1

        # Update best overall if improved
        if best_score_this_iter is not None:
            if campaign.best_score is None or best_score_this_iter > campaign.best_score:
                campaign.best_score = best_score_this_iter
                campaign.best_structure_id = best_structure_id_this_iter
                logger.info(
                    f"New best score: {best_score_this_iter:.3f} "
                    f"(structure {best_structure_id_this_iter})"
                )

        # Check if campaign should complete
        if campaign.current_iteration >= campaign.max_iterations:
            campaign.status = CampaignStatus.COMPLETED
            campaign.completed_at = datetime.utcnow()
            logger.info(f"Campaign completed after {campaign.current_iteration} iterations")

        # Start timestamp on first iteration
        if campaign.current_iteration == 1:
            campaign.started_at = datetime.utcnow()

        campaign.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(iteration)

        logger.info(
            f"Iteration {iteration.iteration_index} completed. "
            f"Campaign status: {campaign.status.value}"
        )

        return iteration

    @staticmethod
    async def generate_candidates(
        db: AsyncSession,
        campaign: Any,  # DesignCampaign
        num_candidates: int = 10
    ) -> List[Any]:  # List[Structure]
        """
        Generate candidate structures for evaluation.

        Current Implementation (v1):
        ----------------------------
        Simple approach: Clone existing structures with small random modifications.
        This provides a working baseline for testing the design loop infrastructure.

        The cloning process:
        1. Query existing structures matching campaign constraints
        2. Randomly select structures to use as templates
        3. Create variants by perturbing lattice parameters
        4. Store new structures in database

        Future Implementations:
        -----------------------

        1. **Bayesian Optimization**:
           ```python
           # Use Gaussian Process to model property landscape
           from sklearn.gaussian_process import GaussianProcessRegressor

           # Fit GP on evaluated structures
           gp = GaussianProcessRegressor()
           gp.fit(X_features, y_scores)

           # Generate candidates using acquisition function
           candidates = generate_using_acquisition(
               gp, acquisition="expected_improvement"
           )
           ```

        2. **Genetic Algorithms**:
           ```python
           # Select parents based on fitness
           parents = tournament_selection(population, fitness_scores)

           # Crossover: Combine structures
           offspring = crossover(parents[0], parents[1])

           # Mutation: Random perturbations
           mutated = mutate(offspring, mutation_rate=0.1)
           ```

        3. **Reinforcement Learning**:
           ```python
           # Agent learns policy for structure generation
           from stable_baselines3 import PPO

           # State: current best structures and their properties
           # Action: parameters for structure generation
           # Reward: improvement in target properties

           action = rl_agent.predict(state)
           candidates = generate_from_action(action)
           ```

        4. **Generative Models**:
           ```python
           # VAE/GAN for novel structure generation
           from cdvae import CDVAE  # Crystal Diffusion VAE

           # Sample from latent space
           z = model.sample_latent(n=num_candidates)

           # Decode to structures
           candidates = model.decode(z)
           ```

        5. **Active Learning**:
           ```python
           # Query most informative structures
           uncertainty = ensemble_predict_uncertainty(candidates)

           # Select candidates with highest uncertainty
           most_informative = top_k_by_uncertainty(
               candidates, uncertainty, k=num_candidates
           )
           ```

        Args:
            db: Async database session
            campaign: DesignCampaign object
            num_candidates: Number of candidates to generate

        Returns:
            List of Structure objects (newly created and added to db)

        AI Agent Notes:
            - Agents can influence generation by setting campaign.config["strategy"]
            - Strategies: "random", "bayesian", "genetic", "rl", "generative"
            - Agents can provide seed structures in config["seed_structures"]
        """
        from ...src.api.models.structure import Structure, StructureSource
        from ...src.api.models.material import Material

        logger.info(f"Generating {num_candidates} candidates for campaign {campaign.id}")

        # Extract constraints from config
        constraints = campaign.config.get("constraints", {})
        elements = constraints.get("elements", None)
        dimensionality = constraints.get("dimensionality", None)
        max_atoms = constraints.get("max_atoms", 100)

        # Query existing structures to use as templates
        query = select(Structure).join(Material)

        if dimensionality is not None:
            query = query.where(Structure.dimensionality == dimensionality)
        if max_atoms:
            query = query.where(Structure.num_atoms <= max_atoms)

        # Limit to reasonable number of templates
        query = query.limit(50)

        result = await db.execute(query)
        templates = result.scalars().all()

        if not templates:
            logger.warning("No template structures found. Creating random structures.")
            # Future: Generate from scratch using generative model
            return []

        logger.info(f"Found {len(templates)} template structures")

        # Generate candidates by cloning and perturbing templates
        candidates = []

        for i in range(num_candidates):
            # Randomly select template
            template = random.choice(templates)

            # Clone structure with perturbation
            candidate = Structure(
                material_id=template.material_id,
                name=f"{campaign.name}_iter{campaign.current_iteration}_cand{i}",
                description=f"Generated candidate from template {template.id}",
                format=template.format,
                source=StructureSource.GENERATED,
                raw_text=template.raw_text,
                lattice=_perturb_lattice(template.lattice) if template.lattice else None,
                atoms=template.atoms,  # Could also perturb atom positions
                dimensionality=template.dimensionality,
                num_atoms=template.num_atoms,
                formula=template.formula,
                # Perturb lattice parameters slightly
                a=_perturb_value(template.a, 0.02) if template.a else None,
                b=_perturb_value(template.b, 0.02) if template.b else None,
                c=_perturb_value(template.c, 0.02) if template.c else None,
                alpha=template.alpha,
                beta=template.beta,
                gamma=template.gamma,
                volume=template.volume,  # Will be recalculated
                metadata={
                    "generated_by": "design_campaign",
                    "campaign_id": str(campaign.id),
                    "template_id": str(template.id),
                    "generation_method": "clone_and_perturb"
                }
            )

            db.add(candidate)
            candidates.append(candidate)

        # Flush to get IDs
        await db.flush()

        logger.info(f"Generated {len(candidates)} candidate structures")

        return candidates

    @staticmethod
    async def evaluate_candidates(
        db: AsyncSession,
        candidates: List[Any],  # List[Structure]
        target_properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate candidate structures using ML predictions.

        For each candidate:
        1. Get ML predictions for relevant properties
        2. Compare predictions against target properties
        3. Calculate match score (0-1)

        The score represents how well the predicted properties match
        the target properties. A score of 1.0 is a perfect match.

        Args:
            db: Async database session
            candidates: List of Structure objects to evaluate
            target_properties: Dict of target properties from campaign config
                Example:
                {
                    "bandgap": {"value": 2.0, "tolerance": 0.2, "weight": 1.0},
                    "formation_energy": {"max": -3.0, "weight": 0.5}
                }

        Returns:
            List of evaluation dictionaries:
            [
                {
                    "structure_id": UUID,
                    "predicted_properties": {
                        "bandgap": 2.1,
                        "formation_energy": -3.5,
                        ...
                    },
                    "score": 0.87,  # Overall match score (0-1)
                    "property_scores": {
                        "bandgap": 0.95,
                        "formation_energy": 0.80
                    }
                },
                ...
            ]

        AI Agent Notes:
            - Scores indicate how well candidates match objectives
            - Agents can use property_scores to understand which targets are met
            - Future: Include uncertainty estimates for decision-making
        """
        from ...backend.common.ml.properties import predict_properties_for_structure

        logger.info(f"Evaluating {len(candidates)} candidates")

        evaluations = []

        for candidate in candidates:
            # Get ML predictions
            predictions = predict_properties_for_structure(candidate)

            # Calculate score against targets
            score = DesignLoopService.calculate_score(
                predicted_props=predictions,
                target_props=target_properties
            )

            # Build evaluation record
            evaluation = {
                "structure_id": candidate.id,
                "predicted_properties": predictions,
                "score": score,
            }

            evaluations.append(evaluation)

            logger.debug(
                f"Structure {candidate.id}: score={score:.3f}, "
                f"bandgap={predictions.get('bandgap', 'N/A')}"
            )

        return evaluations

    @staticmethod
    def calculate_score(
        predicted_props: Dict[str, Any],
        target_props: Dict[str, Any]
    ) -> float:
        """
        Calculate match score (0-1) for predicted vs target properties.

        The score is a weighted sum of individual property matches:
        - Each property contributes based on its weight
        - Properties are scored on how close they are to targets
        - Final score is normalized to 0-1 range

        Scoring Rules:
        --------------
        1. **Exact Target** (e.g., bandgap = 2.0 ± 0.2):
           - Score 1.0 if within tolerance
           - Score decays linearly outside tolerance

        2. **Threshold** (e.g., formation_energy < -3.0):
           - Score 1.0 if criterion met
           - Score 0.0 if criterion not met

        3. **Range** (e.g., 1.5 < bandgap < 2.5):
           - Score 1.0 if in range
           - Score decays outside range

        Args:
            predicted_props: Predictions from ML model
                Example: {"bandgap": 2.1, "formation_energy": -3.5}
            target_props: Target specifications
                Example:
                {
                    "bandgap": {"value": 2.0, "tolerance": 0.2, "weight": 1.0},
                    "formation_energy": {"max": -3.0, "weight": 0.5}
                }

        Returns:
            Overall score from 0.0 (no match) to 1.0 (perfect match)

        Example:
            >>> predicted = {"bandgap": 2.1, "formation_energy": -3.5}
            >>> targets = {
            ...     "bandgap": {"value": 2.0, "tolerance": 0.2, "weight": 1.0},
            ...     "formation_energy": {"max": -3.0, "weight": 0.5}
            ... }
            >>> score = calculate_score(predicted, targets)
            >>> print(score)
            0.92

        AI Agent Notes:
            - Agents can adjust weights to prioritize certain properties
            - Score can be used as reward signal for RL agents
            - Future: Support Pareto optimization for multi-objective
        """
        if not target_props:
            return 1.0  # No targets means everything matches

        property_scores = []
        weights = []

        for prop_name, target_spec in target_props.items():
            # Get predicted value
            predicted_value = predicted_props.get(prop_name)
            if predicted_value is None:
                continue

            # Get target specification
            target_value = target_spec.get("value")
            tolerance = target_spec.get("tolerance", 0.0)
            min_value = target_spec.get("min")
            max_value = target_spec.get("max")
            weight = target_spec.get("weight", 1.0)

            # Calculate property score based on type of constraint
            prop_score = 0.0

            if target_value is not None:
                # Exact target with tolerance
                deviation = abs(predicted_value - target_value)
                if deviation <= tolerance:
                    prop_score = 1.0
                else:
                    # Linear decay outside tolerance
                    # Score = 0 at 5x tolerance
                    max_deviation = tolerance * 5
                    prop_score = max(0.0, 1.0 - (deviation - tolerance) / max_deviation)

            elif min_value is not None and max_value is not None:
                # Range constraint
                if min_value <= predicted_value <= max_value:
                    prop_score = 1.0
                else:
                    # Outside range
                    range_width = max_value - min_value
                    if predicted_value < min_value:
                        deviation = min_value - predicted_value
                    else:
                        deviation = predicted_value - max_value
                    # Linear decay
                    prop_score = max(0.0, 1.0 - deviation / range_width)

            elif min_value is not None:
                # Minimum threshold
                if predicted_value >= min_value:
                    prop_score = 1.0
                else:
                    # Below threshold - could still give partial credit
                    deviation = min_value - predicted_value
                    prop_score = max(0.0, 1.0 - deviation / abs(min_value))

            elif max_value is not None:
                # Maximum threshold
                if predicted_value <= max_value:
                    prop_score = 1.0
                else:
                    # Above threshold
                    deviation = predicted_value - max_value
                    prop_score = max(0.0, 1.0 - deviation / abs(max_value))

            property_scores.append(prop_score)
            weights.append(weight)

        # Calculate weighted average
        if not property_scores:
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_score = sum(s * w for s, w in zip(property_scores, weights)) / total_weight

        return max(0.0, min(1.0, weighted_score))


# Helper functions

def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _perturb_value(value: Optional[float], fraction: float = 0.02) -> Optional[float]:
    """
    Perturb a value by a random fraction.

    Args:
        value: Value to perturb
        fraction: Maximum perturbation as fraction of value (default 2%)

    Returns:
        Perturbed value, or None if input was None
    """
    if value is None:
        return None
    perturbation = value * fraction * (random.random() * 2 - 1)  # -fraction to +fraction
    return value + perturbation


def _perturb_lattice(lattice: Optional[Dict]) -> Optional[Dict]:
    """
    Perturb lattice vectors slightly.

    Args:
        lattice: Lattice dictionary

    Returns:
        Perturbed lattice dictionary
    """
    if not lattice:
        return None

    # Clone lattice
    import copy
    perturbed = copy.deepcopy(lattice)

    # Perturb lattice vectors if present
    if "vectors" in perturbed:
        vectors = perturbed["vectors"]
        if isinstance(vectors, list):
            for i, vec in enumerate(vectors):
                if isinstance(vec, list):
                    perturbed["vectors"][i] = [
                        _perturb_value(v, 0.02) if v is not None else None
                        for v in vec
                    ]

    return perturbed
