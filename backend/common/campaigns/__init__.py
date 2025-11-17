"""
Design campaign service for autonomous materials discovery.

This package provides services for running AI-driven design campaigns,
including structure generation, evaluation, and optimization loops.
"""

from .loop import DesignLoopService

__all__ = [
    "DesignLoopService",
]
