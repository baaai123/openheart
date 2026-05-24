"""
tests/contracts/conftest.py - Shared fixtures and helpers for contract tests.

Contract tests define the interface and behavior of each module.
In RED stage (no implementation), all tests should fail.
In GREEN stage (implementation exists), tests validate actual behavior.
"""

import importlib
import pytest


def require_module(module_path: str, component_name: str):
    """Attempt to import a module. Fail with clear RED STAGE message if absent.

    Every contract test calls this before validating the module's interface.
    In RED stage, this fails because the module doesn't exist yet.
    In GREEN stage, this succeeds and the test validates actual behavior.

    Args:
        module_path: Full dotted import path (e.g. 'src.decision.main_decision')
        component_name: Human-readable name for the RED STAGE message
    """
    try:
        importlib.import_module(module_path)
    except ImportError:
        pytest.fail(
            f"RED STAGE: {component_name} ({module_path}) not yet implemented. "
            f"Contract test validates interface defined in spec v4.5.0."
        )


def fail_red(component_name: str, spec_section: str = ""):
    """Explicit RED-stage failure for behavioral tests that need implementation.

    Use when the test requires running the module, not just importing it.
    """
    section_info = f" (spec {spec_section})" if spec_section else ""
    pytest.fail(
        f"RED STAGE: {component_name} behavior not yet implemented{section_info}. "
        f"Contract test validates the interface defined in spec v4.5.0."
    )
