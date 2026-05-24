"""
RuntimeConfig — immutable configuration object built from environment variables
and GPU detection at system startup.

All mode switches (LOW_VRAM, PERFORMANCE_MODE, SHOW_TRANSCRIPT) are
resolved once when from_environ() is called.
Modules access this via DI or singleton — they must NEVER call os.environ directly.

v4.5.0 §0.5: Runtime Configuration Management
v4.5.0 §12.1: VRAM allocation and two-tier configuration
项目宪法 §3.3: 运行时配置管理
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VRAM tier enum — two-tier design (HIGH / LOW)
# ---------------------------------------------------------------------------

class VRAMTier(str, Enum):
    """
    VRAM configuration tier.

    Two tiers:
      - HIGH: ≥ 12 GB — all models, 4096 context
      - LOW:  < 12 GB — reduced models, 2048 context, CosyVoice on CPU
    """
    HIGH = "high"
    LOW = "low"


# ---------------------------------------------------------------------------
# Exception class
# ---------------------------------------------------------------------------

class SystemRequirementError(RuntimeError):
    """
    Raised when the system does not meet the minimum requirements for OpenHeart.

    v4.5.0 §12.1: if VRAM < 7.5 GB, raise SystemRequirementError and refuse to start.
    """
    pass


# ---------------------------------------------------------------------------
# Environment variable names — the ONLY place these strings appear
# ---------------------------------------------------------------------------

_ENV_LOW_VRAM = "OPENMATE_LOW_VRAM"
_ENV_PERFORMANCE_MODE = "OPENMATE_PERFORMANCE_MODE"
# _ENV_NO_SHADOW and _ENV_ENABLE_SHADOW removed — shadow verification always disabled
_ENV_SHOW_TRANSCRIPT = "OPENMATE_SHOW_TRANSCRIPT"
_ENV_REDIS_HOST = "OPENMATE_REDIS_HOST"
_ENV_REDIS_PORT = "OPENMATE_REDIS_PORT"
_ENV_REDIS_DB = "OPENMATE_REDIS_DB"
_ENV_REDIS_PASSWORD = "OPENMATE_REDIS_PASSWORD"
_ENV_REDIS_AOF = "OPENMATE_REDIS_AOF"
_ENV_DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"

# Redis defaults — spec §3.2.2
_REDIS_HOST_DEFAULT = "localhost"
_REDIS_PORT_DEFAULT = 6379
_REDIS_DB_DEFAULT = 0
_REDIS_AOF_DEFAULT = True

# ---------------------------------------------------------------------------
# Config file paths — resolved from this file's location
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_ENDPOINTS_PATH = _PROJECT_ROOT / "config" / "endpoints.yaml"

# ---------------------------------------------------------------------------
# VRAM thresholds in GB — spec §12.1
# ---------------------------------------------------------------------------

_VRAM_HIGH_THRESHOLD_GB = 12.0
_VRAM_LOW_THRESHOLD_GB = 0.0

# ---------------------------------------------------------------------------
# Context limits — tier-based: HIGH → 4096, LOW → 2048
# ---------------------------------------------------------------------------

_CONTEXT_LIMIT_HIGH = 2048
_CONTEXT_LIMIT_LOW = 2048


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------

def _parse_env_bool(name: str) -> bool | None:
    """
    Parse a boolean environment variable.

    Accepted true values: "1", "true", "yes", "on" (case-insensitive).
    Accepted false values: "0", "false", "no", "off" (case-insensitive).
    Returns None if the variable is not set.

    Args:
        name: Environment variable name.

    Returns:
        Parsed boolean value, or None if not set.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    # v4.5.0 §0.5: unknown values logged as warning, treated as unset
    logger.warning(
        "Unrecognised boolean value for %s: %r. Expected one of: 1/0, true/false, yes/no, on/off. Treating as unset.",
        name, value,
    )
    return None


def _detect_vram_total_gb() -> float:
    """
    Detect total GPU memory on device 0 via torch.cuda.mem_get_info().

    Per task specification: VRAM tier auto-detection uses torch.cuda.mem_get_info().

    Returns:
        Total VRAM in GB as float.

    Raises:
        SystemRequirementError: if CUDA is not available or torch is not installed.
    """
    # Attempt to import torch lazily — only required for GPU detection
    try:
        import torch  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise SystemRequirementError(
            "PyTorch is not installed. OpenHeart requires PyTorch with CUDA support "
            + "and a GPU with CUDA capabilities."
        ) from None

    if not torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
        raise SystemRequirementError(
            "CUDA is not available. OpenHeart requires a CUDA-capable GPU "
            + "with at least 7.5 GB VRAM."
        )

    # mem_get_info returns (free_bytes, total_bytes) for device 0
    # v4.5.0 §3.1: 每次生成前检查 torch.cuda.mem_get_info()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)  # pyright: ignore[reportUnknownMemberType]
    free_gb: float = free_bytes / (1024.0 ** 3)  # pyright: ignore[reportUnknownVariableType]
    vram_total_gb: float = total_bytes / (1024.0 ** 3)  # pyright: ignore[reportUnknownVariableType]

    logger.info(
        "GPU VRAM detected: total=%.1f GB, free=%.1f GB",
        vram_total_gb,
        free_gb,
    )
    return vram_total_gb


def _resolve_vram_tier(vram_total_gb: float, force_low: bool) -> VRAMTier:
    """
    Resolve the VRAM tier from total GPU memory.

    If force_low is True (OPENMATE_LOW_VRAM=1), always returns VRAMTier.LOW.

    Raises SystemRequirementError if VRAM detection returns <= 0.

    Args:
        vram_total_gb: Total GPU VRAM in GB.
        force_low: Whether OPENMATE_LOW_VRAM env var was set.

    Returns:
        Resolved VRAMTier.
    """
    if force_low:
        logger.info("OPENMATE_LOW_VRAM=1: forcing low VRAM tier")
        return VRAMTier.LOW

    if vram_total_gb >= _VRAM_HIGH_THRESHOLD_GB:
        return VRAMTier.HIGH
    elif vram_total_gb > _VRAM_LOW_THRESHOLD_GB:
        return VRAMTier.LOW
    else:
        raise SystemRequirementError(
            f"Detected GPU VRAM ({vram_total_gb:.1f} GB) is below the minimum "
            + "requirement of any supported tier. OpenHeart cannot start."
        )


# ---------------------------------------------------------------------------
# RuntimeConfig — frozen dataclass, immutable after construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeConfig:
    """
    Immutable runtime configuration resolved once at system startup.

    All mode switches (LOW_VRAM, PERFORMANCE_MODE, NO_SHADOW, ENABLE_SHADOW,
    SHOW_TRANSCRIPT) are read from environment variables and converted to
    typed fields in this object. Modules access config via dependency injection
    or the global singleton — they must NEVER call os.environ directly.

    v4.5.0 §0.5: Runtime Configuration Management
    项目宪法 §3.3: 禁止在模块内部直接读取环境变量

    Attributes:
        vram_tier: The auto-detected VRAM configuration tier.
        vram_total_gb: Total GPU VRAM in GB (float).
        low_vram: True if running in low-VRAM mode (tier LOW or forced by env).
        performance_mode: True if OPENMATE_PERFORMANCE_MODE is set.
        enable_shadow: Always False — shadow verification removed.
        show_transcript: True if transcript overlay window should be shown.
            Default True per spec §7.5.6.
        context_limit: Maximum token context limit.
            HIGH tier: 4096, LOW tier: 2048.
    """

    # ---- VRAM configuration ----
    vram_tier: VRAMTier
    """Auto-detected VRAM tier (HIGH or LOW)."""

    vram_total_gb: float
    """Total GPU VRAM in GB, as detected from torch.cuda.mem_get_info()."""

    low_vram: bool
    """True if running in low-VRAM mode, either auto-detected or forced."""

    # ---- Mode switches ----
    performance_mode: bool
    """True if OPENMATE_PERFORMANCE_MODE env var is set."""

    enable_shadow: bool
    """Always False — shadow verification has been removed."""

    show_transcript: bool
    """True if transcript overlay window should be displayed.
    v4.5.0 §7.5.6: Default True. Can be toggled via OPENMATE_SHOW_TRANSCRIPT."""

    # ---- Redis configuration ----
    redis_host: str
    """Redis server hostname (default: localhost)."""

    redis_port: int
    """Redis server port (default: 6379)."""

    redis_db: int
    """Redis database number (default: 0)."""

    redis_password: str | None
    """Redis password (default: None, no authentication)."""

    redis_aof: bool
    """Whether AOF persistence is expected on the Redis side."""

    # ---- DeepSeek cloud API configuration ----
    deepseek_api_key: str
    """DeepSeek API key, read from DEEPSEEK_API_KEY env var at startup."""

    deepseek_base_url: str
    """DeepSeek API base URL (default: https://api.deepseek.com/v1)."""

    deepseek_model: str
    """DeepSeek model name (default: deepseek-v4-flash)."""

    deepseek_max_tokens: int
    """Maximum tokens per DeepSeek API request (default: 200)."""

    deepseek_temperature: float
    """Temperature for DeepSeek API generation (default: 0.8)."""

    # ---- Context configuration ----
    context_limit: int
    """Maximum token count for the assembled context window.
    v4.5.0 §0.4: Default 2048 tokens. Performance mode: 4096 tokens."""

    # ------------------------------------------------------------------
    # Factory method — the ONLY place os.environ is read
    # ------------------------------------------------------------------

    @classmethod
    def from_environ(cls) -> RuntimeConfig:
        """
        Create a RuntimeConfig from environment variables and GPU detection.

        This is the SINGLE entry point for reading environment variables.
        Call once at system startup. All subsequent config access goes through
        the returned frozen object.

        Environment variables read (all prefixed OPENMATE_):
            LOW_VRAM          — force low VRAM tier (1/true/yes/on)
            PERFORMANCE_MODE  — enable performance mode
            SHOW_TRANSCRIPT   — show/hide transcript overlay window
            REDIS_HOST        — Redis hostname (default: localhost)
            REDIS_PORT        — Redis port (default: 6379)
            REDIS_DB          — Redis database number (default: 0)
            REDIS_PASSWORD    — Redis password (default: none)
            REDIS_AOF         — expect AOF persistence (default: true)

        Non-OPENMATE_ env vars:
            DEEPSEEK_API_KEY  — DeepSeek API key (also read from config/endpoints.yaml)

        VRAM tier auto-detection (two-tier):
            HIGH: ≥ 12 GB (all models, 4096 context)
            LOW:  < 12 GB (reduced models, 2048 context, CosyVoice on CPU)

        Returns:
            An immutable RuntimeConfig instance.

        Raises:
            SystemRequirementError: If system does not meet minimum requirements.
        """
        # ---- Read ALL environment variables ONCE ----
        force_low_vram: bool = _parse_env_bool(_ENV_LOW_VRAM) is True
        performance_mode: bool = _parse_env_bool(_ENV_PERFORMANCE_MODE) is True
        show_transcript_env: bool | None = _parse_env_bool(_ENV_SHOW_TRANSCRIPT)

        # Redis config — v4.5.0 §3.2.2
        redis_host: str = os.environ.get(_ENV_REDIS_HOST, _REDIS_HOST_DEFAULT)
        redis_port_raw: str | None = os.environ.get(_ENV_REDIS_PORT)
        redis_db_raw: str | None = os.environ.get(_ENV_REDIS_DB)
        redis_password: str | None = os.environ.get(_ENV_REDIS_PASSWORD) or None
        redis_aof: bool = _parse_env_bool(_ENV_REDIS_AOF) is not False  # default True

        try:
            redis_port: int = int(redis_port_raw) if redis_port_raw else _REDIS_PORT_DEFAULT
        except (ValueError, TypeError):
            logger.warning("Invalid OPENMATE_REDIS_PORT=%r, using default %d", redis_port_raw, _REDIS_PORT_DEFAULT)
            redis_port = _REDIS_PORT_DEFAULT

        try:
            redis_db: int = int(redis_db_raw) if redis_db_raw else _REDIS_DB_DEFAULT
        except (ValueError, TypeError):
            logger.warning("Invalid OPENMATE_REDIS_DB=%r, using default %d", redis_db_raw, _REDIS_DB_DEFAULT)
            redis_db = _REDIS_DB_DEFAULT

        # ---- DeepSeek cloud API ----
        deepseek_api_key: str = os.environ.get(_ENV_DEEPSEEK_API_KEY, "")
        deepseek_base_url: str = "https://api.deepseek.com/v1"
        deepseek_model: str = "deepseek-v4-flash"
        deepseek_max_tokens: int = 200
        deepseek_temperature: float = 0.8
        try:
            import yaml  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
            with open(_ENDPOINTS_PATH, "r") as f:
                _endpoints_data: dict = yaml.safe_load(f) or {}
            _ds: dict = _endpoints_data.get("deepseek", {})
            if _ds:
                deepseek_base_url = str(_ds.get("base_url", deepseek_base_url))
                deepseek_model = str(_ds.get("model", deepseek_model))
                deepseek_max_tokens = int(_ds.get("max_tokens", deepseek_max_tokens))
                deepseek_temperature = float(_ds.get("temperature", deepseek_temperature))
                # Resolve api_key from YAML if env var is empty
                if not deepseek_api_key:
                    _yaml_key = str(_ds.get("api_key", ""))
                    if _yaml_key.startswith("${") and _yaml_key.endswith("}"):
                        _env_name = _yaml_key[2:-1]
                        deepseek_api_key = os.environ.get(_env_name, "")
                    else:
                        deepseek_api_key = _yaml_key
            else:
                logger.warning("No 'deepseek' section in %s, using defaults", _ENDPOINTS_PATH)
        except Exception as exc:
            # YAML not available or file not readable — use defaults, log warning
            logger.warning("Could not load %s: %s, using defaults", _ENDPOINTS_PATH, exc)

        # ---- Auto-detect VRAM ----
        vram_total_gb: float = _detect_vram_total_gb()
        vram_tier: VRAMTier = _resolve_vram_tier(vram_total_gb, force_low_vram)

        # ---- Derived fields ----
        low_vram: bool = (vram_tier == VRAMTier.LOW) or force_low_vram

        # Shadow verification — always disabled
        enable_shadow: bool = False

        # Transcript overlay — default True (§7.5.6)
        show_transcript: bool = (
            show_transcript_env if show_transcript_env is not None else True
        )

        # Context limit — tier-based: HIGH → 4096, LOW → 2048
        context_limit: int = (
            _CONTEXT_LIMIT_HIGH if vram_tier == VRAMTier.HIGH
            else _CONTEXT_LIMIT_LOW
        )

        config = cls(
            vram_tier=vram_tier,
            vram_total_gb=vram_total_gb,
            low_vram=low_vram,
            performance_mode=performance_mode,
            enable_shadow=enable_shadow,
            show_transcript=show_transcript,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            redis_password=redis_password,
            redis_aof=redis_aof,
            deepseek_api_key=deepseek_api_key,
            deepseek_base_url=deepseek_base_url,
            deepseek_model=deepseek_model,
            deepseek_max_tokens=deepseek_max_tokens,
            deepseek_temperature=deepseek_temperature,
            context_limit=context_limit,
        )

        logger.info(
            "RuntimeConfig initialised: tier=%s, vram=%.1fGB, "
            + "perf=%s, transcript=%s, redis=%s:%d/%d, aof=%s, context=%d, "
            + "deepseek=%s/%s",
            vram_tier.value,
            vram_total_gb,
            performance_mode,
            show_transcript,
            redis_host,
            redis_port,
            redis_db,
            redis_aof,
            context_limit,
            deepseek_model,
            deepseek_base_url,
        )

        return config
