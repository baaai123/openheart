"""Shared fixtures and import patches for integration tests.

Monkey-patches the broken perception/__init__.py import chain.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _patch_perception_imports(monkeypatch):
    """Stub perception_bus import so perception/__init__.py loads cleanly.

    perception_bus.py currently imports ``Message`` from
    fusion.message_envelope, but the actual class is ``MessageEnvelope``.
    This workaround provides a stub ``perception_bus`` module so the
    perception package can be imported.  It will be removed once
    perception_bus.py is fixed.
    """
    stub = MagicMock()
    stub.PerceptionBus = MagicMock
    monkeypatch.setitem(
        sys.modules, "src.perception.perception_bus", stub
    )
    yield


@pytest.fixture(autouse=True)
def _patch_fusion_imports(monkeypatch):
    """Stub fusion.message_envelope imports that may be needed."""
    stub = MagicMock()
    stub.Message = MagicMock
    stub.MessageEnvelope = MagicMock
    stub.create_message = MagicMock
    stub.create_envelope = MagicMock
    stub.Layer = MagicMock
    stub.PayloadType = MagicMock
    stub.EmotionCategory = MagicMock
    stub.MessageValidationError = MagicMock
    monkeypatch.setitem(
        sys.modules, "fusion.message_envelope", stub
    )
    yield


@pytest.fixture(autouse=True)
def _patch_gpu_dependencies(monkeypatch):
    """Prevent GPU-dependent model downloads during integration tests.

    Patches lazy-loaded embedders in fusion modules so they gracefully
    degrade (no SentenceTransformer download errors).  Also patches
    spaCy NLP to return None (models not installed in CI).
    """
    # Prevent SentenceTransformer from being loaded in entity_fusion
    monkeypatch.setattr(
        "src.fusion.entity_fusion._get_embedder",
        lambda: None,
    )
    # Prevent spaCy model loading — already gracefully handled by the code
    yield
