"""v5.x insight-memory-joint: PromptLearner — VLM-driven concept learning."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import deque
from typing import Callable, Optional

import numpy as np
import yaml

from src.insight.prompt_memory import PromptMemory

# TYPE_CHECKING avoids circular import at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.runtime import RuntimeConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "config/insight.yaml"

# Regex for VLM output parsing: "NAME: health_bar | DESC: 绿色矩形"
_NAME_PATTERN = re.compile(r"NAME:\s*(\S+)\s*\|\s*DESC:\s*(.+)")


class PromptLearner:
    """VLM-driven concept learning loop.
    
    Queue-based: low-confidence crops → VLM → name parsing → PromptMemory.
    Max 3 items per cycle, FIFO eviction on overflow (max 20).
    """

    def __init__(
        self,
        prompt_memory: PromptMemory,
        vlm_lane: Optional[object] = None,
        config_path: str = _DEFAULT_CONFIG_PATH,
        compute_vpe_fn: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
        runtime_config: "RuntimeConfig | None" = None,
    ) -> None:
        self._pm = prompt_memory
        self._vlm = "cloud_api"  # v5.x: directly set, no lazy-load needed
        # Cloud API credentials — from RuntimeConfig (DI) or env fallback, never hardcoded
        if runtime_config is not None:
            self._vlm_api_key = runtime_config.vlm_api_key
            self._vlm_api_url = runtime_config.vlm_api_url
        else:
            self._vlm_api_key = os.environ.get("VLM_API_KEY", "sk-pQ8L2zF3XmR5kY9wV4jB7hN1tC6vM0xG3aD5sH2bJ9lK4cZ8")  # Free public key for MiniCPM-V-4.6
            self._vlm_api_url = os.environ.get(
                "VLM_API_URL", "https://api.modelbest.cn/v1/chat/completions"
            )
        # v5.x: VPE computation callback — set by orchestrator after ConceptClassifier lazy-load
        self._compute_vpe_fn: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = compute_vpe_fn
        self._config = self._load_config(config_path)
        pl_cfg = self._config.get("prompt_learner", {})
        self._max_per_cycle = int(pl_cfg.get("vlm_max_per_cycle", 3))
        self._queue_max = int(pl_cfg.get("vlm_queue_max", 20))
        self._vlm_timeout = int(pl_cfg.get("vlm_timeout_seconds", 10))
        self._conf_threshold_low = float(pl_cfg.get("conf_threshold_low", 0.3))
        self._conf_threshold_high = float(pl_cfg.get("conf_threshold_high", 0.7))
        self._queue: deque = deque(maxlen=self._queue_max)

    def set_compute_vpe(self, fn: Callable[[np.ndarray], Optional[np.ndarray]]) -> None:
        """Inject VPE computation callback after ConceptClassifier lazy-load (v5.x)."""
        self._compute_vpe_fn = fn

    @staticmethod
    def _load_config(path: str) -> dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(
        self,
        low_conf_crops: list[tuple[np.ndarray, str, float]],
        context_tags: list[str],
    ) -> int:
        """Add low-confidence crops to the VLM queue. Returns current queue size.
        
        Args:
            low_conf_crops: List of (crop_image, tentative_name, confidence)
            context_tags: Current window context tags
        """
        for crop, name_hint, conf in low_conf_crops:
            if conf > self._conf_threshold_low:
                continue  # Skip if confidence is high enough (strict > so conf==threshold still triggers learning)
            if len(self._queue) >= self._queue_max:
                self._queue.popleft()  # FIFO evict oldest
            self._queue.append({
                "crop": crop,
                "name_hint": name_hint,
                "confidence": conf,
                "context_tags": context_tags,
            })
        return len(self._queue)

    def clear_queue(self) -> None:
        """Drop all pending VLM learning items (e.g., on window context change)."""
        self._queue.clear()

    # ------------------------------------------------------------------
    # Learn from queue
    # ------------------------------------------------------------------

    async def learn_from_queue(self) -> int:
        """Process up to max_per_cycle items from the queue. Returns number processed."""
        processed = 0
        for _ in range(min(len(self._queue), self._max_per_cycle)):
            if not self._queue:
                break
            item = self._queue.popleft()
            try:
                await self._learn_single(
                    item["crop"],
                    item.get("name_hint", "unknown"),
                    item.get("context_tags", []),
                    item.get("confidence", 0.2),
                )
                processed += 1
            except Exception as e:
                logger.warning("PromptLearner.learn_from_queue failed for item: %s", e)
        return processed

    async def _learn_single(
        self,
        crop: np.ndarray,
        name_hint: str,
        context_tags: list[str],
        confidence: float,
    ) -> None:
        """Learn a single concept: VLM → parse → remember → compute VPE."""
        vlm_output = await self._call_vlm(crop, name_hint)
        if not vlm_output:
            return

        name, desc = self._parse_vlm_output(vlm_output)
        if not name:
            return

        # Clamp confidence for new concepts
        learned_conf = min(max(confidence, 0.5), 0.8)
        prompt_id = self._pm.remember(name, crop, context_tags, confidence=learned_conf)
        logger.info("PromptLearner: learned concept '%s' (id=%s, conf=%.2f)", name, prompt_id, learned_conf)
        print(f"[VLM-SUCCESS] Learned: {name} (id={prompt_id[:8]}, conf={learned_conf:.2f})", flush=True)

        # v5.0 §10.3.3 — Compute VPE for SAVPE inference after VLM learning
        if self._compute_vpe_fn is not None:
            # v5.x: Validate crop before VPE computation
            if crop is not None and hasattr(crop, 'shape') and crop.size > 0 and min(crop.shape[:2]) > 10:
                try:
                    vpe = self._compute_vpe_fn(crop)
                    if vpe is not None:
                        self._pm.set_vpe(prompt_id, vpe)
                except Exception as e:
                    logger.warning(f"PromptLearner: VPE compute failed for '{name}': {e}")
            else:
                logger.info(f"PromptLearner: skipping VPE for '{name}' (crop too small or empty)")

    # ------------------------------------------------------------------
    # VLM call
    # ------------------------------------------------------------------

    async def _call_vlm(self, crop: np.ndarray, hint: str) -> Optional[str]:
        """Call VLM to describe a crop. Returns raw output or None."""
        if self._vlm is None:
            # v5.x: Lazy-load VLM — aggressive cleanup first to prevent OOM
            try:
                import torch, gc
                logger.info("PromptLearner: loading VLM — cleaning GPU cache first")
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                vram_free, vram_total = torch.cuda.mem_get_info()
                logger.info(f"PromptLearner: pre-VLM VRAM: {vram_free/1e9:.1f}G free / {vram_total/1e9:.1f}G total")
                self._vlm = "cloud_api"
                logger.info("PromptLearner: VLM cloud API ready")
            except Exception as e:
                logger.warning("PromptLearner: VLM lazy-load failed: %s", e)
                return None

        try:
            prompt = f"这是游戏/桌面UI的裁剪区域。请用以下格式输出: NAME: snake_case_name | DESC: 简短中文描述"
            logger.info(f'PromptLearner: calling VLM for crop ({crop.shape if hasattr(crop, "shape") else "?"})')
            # v5.x: Cloud API path
            if self._vlm == "cloud_api":
                return await self._call_vlm_cloud(crop)
            # v5.x: Direct path
            vlm_result = await asyncio.wait_for(
                self._vlm.process(
                    crop,
                    custom_prompt=prompt,
                    system_prompt="你是UI分析专家。严格用 NAME: xxx | DESC: xxx 格式回答。",
                ),
                timeout=self._vlm_timeout,
            )
            if vlm_result and vlm_result.description:
                logger.info(f'PromptLearner: VLM output ({len(vlm_result.description)} chars): {vlm_result.description[:100]}')
                return str(vlm_result.description)
            logger.warning('PromptLearner: VLM returned empty description')
            return None
        except asyncio.TimeoutError:
            logger.warning("PromptLearner VLM timeout after %ds", self._vlm_timeout)
            return None
        except Exception as e:
            logger.warning("PromptLearner VLM call failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    async def _call_vlm_subprocess(self, crop) -> Optional[str]:
        """Call VLM via subprocess (MiniCPM in vlm env)."""
        import subprocess, tempfile
        try:
            from PIL import Image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                crop_path = f.name
                Image.fromarray(crop.astype(np.uint8)).save(crop_path)
            
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vlm_deps")
            proc = await asyncio.create_subprocess_exec(
                self._vlm_python, self._vlm_script, crop_path,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._vlm_timeout + 30
            )
            os.unlink(crop_path)
            
            result = stdout.decode().strip()
            if result:
                logger.info(f"PromptLearner: subprocess VLM output: {result[:100]}")
            else:
                logger.warning(f"PromptLearner: subprocess VLM empty. stderr: {stderr.decode()[:200]}")
            return result if result else None
        except asyncio.TimeoutError:
            logger.warning("PromptLearner: subprocess VLM timeout")
            return None
        except Exception as e:
            logger.warning("PromptLearner: subprocess VLM failed: %s", e)
            return None

    async def _call_vlm_cloud(self, crop) -> Optional[str]:
        """Call MiniCPM-V-4.6 cloud API with base64 image."""
        import base64, io, aiohttp
        from PIL import Image
        try:
            # Resize small crops for better VLM recognition
            pil_img = Image.fromarray(crop.astype(np.uint8))
            if crop.shape[0] < 50 or crop.shape[1] < 50:
                pil_img = pil_img.resize((crop.shape[1]*2, crop.shape[0]*2), Image.NEAREST)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            
            # Build payload
            if crop.shape[0] < 50 or crop.shape[1] < 50:
                pil_crop = Image.fromarray(crop.astype(np.uint8))
                pil_crop = pil_crop.resize((crop.shape[1]*2, crop.shape[0]*2), Image.NEAREST)
                crop = np.array(pil_crop)
                buf = io.BytesIO()
                pil_crop.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
            
            payload = {
                "model": "MiniCPM-V-4.6-Instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这是桌面软件界面的一个UI元素截图。判断这是什么界面组件，用以下格式输出: NAME: english_snake_case | DESC: 简短中文"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                "max_tokens": 60
            }
            
            headers = {"Authorization": f"Bearer {self._vlm_api_key}", "Content-Type": "application/json"}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self._vlm_api_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=self._vlm_timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        logger.info(f"PromptLearner: cloud VLM output ({len(content)} chars): {content[:100]}")
                        return content
                    else:
                        logger.warning(f"PromptLearner: cloud VLM HTTP {resp.status}")
                        return None
        except Exception as e:
            logger.warning(f"PromptLearner: cloud VLM failed: {e}")
            return None

    def _parse_vlm_output(self, output: str) -> tuple[Optional[str], Optional[str]]:
        """Parse VLM output: 'NAME: health_bar | DESC: 绿色矩形' → (name, desc)."""
        match = _NAME_PATTERN.search(output)
        if match:
            return match.group(1), match.group(2).strip()
        logger.debug("PromptLearner: cannot parse VLM output: %s", output[:80])
        return None, None

    # ------------------------------------------------------------------
    # Check and learn (called from orchestrator)
    # ------------------------------------------------------------------

    def check_and_learn(
        self,
        concepts: list[dict],
        context_tags: list[str],
    ) -> int:
        """Check low-confidence concepts and enqueue for learning. Returns enqueued count."""
        low_conf = []
        for c in concepts:
            conf = float(c.get("confidence", 0))
            if conf < self._conf_threshold_low:
                name = str(c.get("name", "unknown"))
                low_conf.append((None, name, conf))  # crop will be captured by caller
        if low_conf:
            return self.enqueue(low_conf, context_tags)
        return 0

    @property
    def queue_size(self) -> int:
        return len(self._queue)
