# tests/contracts/__init__.py - Contract tests for OpenHeart modules
# v4.5.0 - Each module must pass its corresponding contract test before being considered "done".

import importlib
import pytest


def require_module(module_path: str, component_name: str):
    try:
        importlib.import_module(module_path)
    except ImportError:
        pytest.fail(
            f"RED STAGE: {component_name} ({module_path}) not yet implemented. "
            f"Contract test validates interface defined in spec v4.5.0."
        )


def fail_red(component_name: str, spec_section: str = ""):
    section_info = f" (spec {spec_section})" if spec_section else ""
    pytest.fail(
        f"RED STAGE: {component_name} behavior not yet implemented{section_info}. "
        f"Contract test validates the interface defined in spec v4.5.0."
    )
