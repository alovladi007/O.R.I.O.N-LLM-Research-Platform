"""
ORION Platform — top-level package.

This __init__ intentionally avoids importing legacy submodules
(src.core, src.knowledge_graph, src.rag, etc.) so that the canonical
backend (src.api.app) can be imported without dragging in the pre-refactor
codebase.

Legacy modules are scheduled for removal or integration in Phase 0 /
Session 0.2 (see ROADMAP_PROMPTS.md).
"""

__version__ = "1.0.0"
__author__ = "ORION Development Team"
__license__ = "MIT"
