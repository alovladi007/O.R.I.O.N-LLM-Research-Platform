"""Default tool catalog for the agent loop.

Ships the four roadmap-named tools (``structure_generator``,
``run_dft``, ``run_ml_predict``, ``suggest_bo``) with **stub
handlers**: the schemas are real, the handlers return mock data
shaped like the real responses. Session 7.3b replaces the handlers
with calls into the real engines (BO from Session 7.1, DFT job
dispatch from Phase 3 / Session 2.x, ML predict from Phase 6 / 6.4).

Cost estimates
--------------

The cost-estimate functions matter even for the stubs — the cost
guard's "max_cost_usd=0 halts before the first DFT submission"
acceptance test requires that ``run_dft`` reports a non-zero
estimate. Numbers below are rough cluster-rate ballparks; tune in
7.3b once we have real wall-time data.
"""

from __future__ import annotations

from typing import Any, Dict

from .agent import Tool, ToolCatalog


# ---------------------------------------------------------------------------
# Stub handlers — each returns a dict shaped like the eventual real call
# ---------------------------------------------------------------------------


def _stub_structure_generator(args: Dict[str, Any]) -> Dict[str, Any]:
    """Mock: generate ``n`` candidate compositions from a prototype."""
    n = int(args.get("n_candidates", 1))
    prototype = args.get("prototype", "ABO3")
    elements = args.get("elements", [])
    return {
        "candidates": [
            {
                "structure_id": f"stub-{prototype}-{i}",
                "composition_formula": f"{prototype}_{i}",
                "elements": list(elements),
            }
            for i in range(n)
        ],
        "via": "stub",
    }


def _stub_run_dft(args: Dict[str, Any]) -> Dict[str, Any]:
    """Mock: simulate a DFT static calc on a structure."""
    return {
        "job_id": f"stub-dft-{args.get('structure_id', '?')}",
        "status": "succeeded",
        "outputs": {
            "total_energy_eV_per_atom": -7.5,
            "bandgap_eV": 2.0,
            "formation_energy_eV_per_atom": -1.2,
        },
        "via": "stub",
    }


def _stub_run_ml_predict(args: Dict[str, Any]) -> Dict[str, Any]:
    """Mock: ML prediction with uncertainty for the requested property."""
    prop = args.get("property", "bandgap")
    return {
        "structure_id": args.get("structure_id", "?"),
        "property": prop,
        "value": 2.1,
        "sigma": 0.4,
        "via": "stub",
    }


def _stub_suggest_bo(args: Dict[str, Any]) -> Dict[str, Any]:
    """Mock: BO suggestion in the same shape Session 7.1 returns."""
    q = int(args.get("q", 1))
    return {
        "candidates": [
            {"x1": float(0.1 * i), "x2": float(0.5 * i)} for i in range(q)
        ],
        "via": "stub",
    }


# ---------------------------------------------------------------------------
# Cost estimators
# ---------------------------------------------------------------------------


def _cost_structure_generator(args: Dict[str, Any]) -> float:
    return 0.0  # local CPU only


def _cost_run_dft(args: Dict[str, Any]) -> float:
    """Static-calc default: ~$0.50/run on a small cluster.

    A relax is more like $5; tune once the real Phase-3 dispatcher
    feeds wall-time back. The acceptance test only depends on this
    being **strictly positive** so the cost guard at $0 trips.
    """
    if args.get("calculation", "static") == "relax":
        return 5.0
    return 0.5


def _cost_run_ml_predict(args: Dict[str, Any]) -> float:
    return 0.0


def _cost_suggest_bo(args: Dict[str, Any]) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# Default catalog
# ---------------------------------------------------------------------------


def default_tool_catalog() -> ToolCatalog:
    """Build a fresh :class:`ToolCatalog` with the four roadmap tools.

    Caller can ``register`` more tools after this returns; the four
    base tools cover the agent's standard plan (generate → predict
    → suggest → confirm with DFT).
    """
    cat = ToolCatalog()
    cat.register(Tool(
        name="structure_generator",
        description=(
            "Generate candidate structures from a chemical prototype "
            "and an element set. Use this to seed a campaign."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prototype": {
                    "type": "string",
                    "description": (
                        "Crystal prototype (e.g. 'ABO3' for perovskite, "
                        "'AB2' for fluorite)."
                    ),
                },
                "elements": {
                    "type": "array",
                    "description": "Element symbols allowed in the candidates.",
                },
                "n_candidates": {
                    "type": "integer",
                    "description": "Number of candidates to generate.",
                    "default": 5,
                },
            },
            "required": ["prototype", "elements"],
        },
        handler=_stub_structure_generator,
        cost_estimate_usd=_cost_structure_generator,
    ))
    cat.register(Tool(
        name="run_dft",
        description=(
            "Submit a DFT calculation (static or relax) for a structure. "
            "Returns the job id and the parsed outputs once finished. "
            "Expensive — use only when ML uncertainty is too high."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "structure_id": {
                    "type": "string",
                    "description": "ORION structure UUID or external id.",
                },
                "calculation": {
                    "type": "string",
                    "description": "One of 'static', 'relax'.",
                    "default": "static",
                },
                "functional": {
                    "type": "string",
                    "description": "XC functional. Default 'PBE'.",
                    "default": "PBE",
                },
            },
            "required": ["structure_id"],
        },
        handler=_stub_run_dft,
        cost_estimate_usd=_cost_run_dft,
    ))
    cat.register(Tool(
        name="run_ml_predict",
        description=(
            "ML prediction (mu, sigma) for a target property on a "
            "structure. Cheap; use as the first pass before falling "
            "through to DFT."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string"},
                "property": {
                    "type": "string",
                    "description": (
                        "One of 'bandgap', 'formation_energy', "
                        "'bulk_modulus', 'density'."
                    ),
                    "default": "bandgap",
                },
            },
            "required": ["structure_id"],
        },
        handler=_stub_run_ml_predict,
        cost_estimate_usd=_cost_run_ml_predict,
    ))
    cat.register(Tool(
        name="suggest_bo",
        description=(
            "Ask the BO engine for the next q candidate parameter "
            "vectors given the current campaign history."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "campaign_id": {"type": "string"},
                "q": {"type": "integer", "default": 1},
            },
            "required": ["campaign_id"],
        },
        handler=_stub_suggest_bo,
        cost_estimate_usd=_cost_suggest_bo,
    ))
    return cat
