"""Infrastructure layer — logging, config, GPU safety, Docker lifecycle, shutdown.

v4.5.0 §10.1: JSON logging with trace_id propagation.
v4.5.0 §10.2: Distributed tracing via TraceManager + @trace_span.
v4.5.0 §12.2: GPU inference thread with model_load_semaphore.
v4.5.0 §3.2: Docker Compose for Redis (AOF) + LanceDB persistence.
"""
from .logging_setup import setup_logging, trace_id_var, set_trace_id
from .config_validator import ConfigValidator, ValidationReport, ValidationIssue
from .gpu_manager import GPUInferenceThread, get_gpu_thread
from .docker_manager import DockerManager, get_docker_manager
from .shutdown_manager import ShutdownManager, get_shutdown_manager
from .tracing import (
    SPAN_DEGRADED,
    SPAN_FAILURE,
    SPAN_SUCCESS,
    SPAN_TIMEOUT,
    TraceManager,
    generate_trace_id,
    sync_trace_span,
    trace_span,
)

__all__ = [
    "setup_logging", "trace_id_var", "set_trace_id",
    "ConfigValidator", "ValidationReport", "ValidationIssue",
    "GPUInferenceThread", "get_gpu_thread",
    "DockerManager", "get_docker_manager",
    "ShutdownManager", "get_shutdown_manager",
    "TraceManager", "trace_span", "sync_trace_span", "generate_trace_id",
    "SPAN_SUCCESS", "SPAN_FAILURE", "SPAN_DEGRADED", "SPAN_TIMEOUT",
]
