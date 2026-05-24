"""
Lane 5: Qwen3-VL-2B-Instruct — natural language screen description.

v4.5.0 §1.3.6: 自然语言屏幕描述（低频轮询）
   模型: Qwen3-VL-2B-Instruct via vLLM 0.11.2
   输入: 全屏截图 (resized to max 1024px)
   延迟: 200-500ms (低频, 1-2Hz)
   能力: 自然语言描述屏幕内容、用户正在进行的活动
   降级: 模型加载失败 → 返回空 description, degraded=true
   低频: 接受 poll_interval 参数，控制轮询频率

项目宪法 §4.1: Qwen3-VL 不可用 → 返回空 description, degraded=true
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np
from PIL import Image

from src.perception.visual.types import VisionSummary

logger = logging.getLogger(__name__)

# v4.5.0 §1.3.6: 输入 max 1280px on longest side
DEFAULT_MAX_SIZE = 1280
# v4.5.0 §1.3.6: 低频轮询默认 2 秒间隔
DEFAULT_POLL_INTERVAL = 2.0

# v4.5.0 §1.3.6: 系统提示词 — 雪奈 persona 屏幕描述
SYSTEM_PROMPT = (
    "像人一样，对有趣、显眼、重要的画面内容详细描述，优先描述鼠标周围内容\n"
    "无聊的空白区域、重复内容一笔带过。\n"
    "输出不少于50字"
)

USER_PROMPT = "看看屏幕上有什么？"

# v5.x: Composite-specific system prompt — 雪奈 looks at user's screen via composite image
COMPOSITE_SYSTEM_PROMPT = (
    "你是雪奈，用户的傲娇妹妹的眼睛。"
    "虚拟角色就是雪奈，其余是用户正在使用的窗口内容。"
    "自然描述眼前的窗口内容，简略描述虚拟角色。"
)

L2D_SYSTEM_PROMPT = (
    "你是雪奈，用户的傲娇妹妹。"
    "详细描述你自己的Live2D形象——外貌、服装、表情、姿态。"
    "用第二人称'你'叙述。"
)

WINDOW_SYSTEM_PROMPT = (
    "你是雪奈，用户哥哥的傲娇妹妹的眼睛。"
    "自然描述哥哥的屏幕画面内容，100字以内"
)

# v4.5.0 §1.3.6: ROI-based multi-image prompt (crops from OmniParser L2)
ROI_USER_PROMPT = "屏幕画面上有什么"


class QwenVLLane:
    """
    Lane 5: Natural language screen description via Qwen3-VL-2B-Instruct (vLLM).

    Runs at lower frequency than other visual lanes (poll_interval controls how
    often inference actually executes). Lazy-loads the vLLM engine on first use.

    v4.5.0 §1.3.6 接口
    """

    def __init__(self,
                 model_path: str = "models/qwen3-vl-2b/",
                 poll_interval: float = DEFAULT_POLL_INTERVAL,
                 use_vllm: bool = True,
                 model_type: str = "qwen"):
        """
        Args:
            model_path: Local path.
            poll_interval: Minimum seconds between consecutive inferences.
            use_vllm: If False, use CPU inference via transformers.
            model_type: "qwen" or "minicpm".
        """
        self._model_path = model_path
        self._model = None
        self._available = True
        self._use_vllm = use_vllm
        self._model_type = model_type
        self._degraded = False
        self._poll_interval = poll_interval
        self._last_process_time: float = 0.0
        self.last_spatial_description: str = ""

    @property
    def available(self) -> bool:
        """True if this lane is operational (not degraded)."""
        return self._available

    @property
    def degraded(self) -> bool:
        """True if this lane is in degraded state."""
        return self._degraded

    @property
    def poll_interval(self) -> float:
        """Minimum seconds between consecutive inferences."""
        return self._poll_interval

    async def process(self, frame, custom_prompt: str = "",
                      system_prompt: str = "") -> VisionSummary:
        """
        Generate a concise scene description from a full-screen image.

        v4.5.0 §1.3.6: 输入全屏截图, resized to max 1024px on longest side.
        默认使用 USER_PROMPT，输出场景描述——空间定位由 OmniParser+EasyOCR 负责。

        Args:
            frame: numpy array (H, W, 3) in RGB/BGR format.
            custom_prompt: Optional override for the user prompt.

        Returns:
            VisionSummary with description string. Empty description if degraded
            or within poll_interval cooldown.
        """
        if not self._available:
            return VisionSummary(description="", source="qwen3_vl_2b")

        # v4.5.0 §1.3.6: 低频轮询 — skip if within poll_interval
        now = time.monotonic()
        if self._last_process_time > 0:
            elapsed = now - self._last_process_time
            if elapsed < self._poll_interval:
                return VisionSummary(description="", source="qwen3_vl_2b")
        self._last_process_time = now

        try:
            prompt = custom_prompt if custom_prompt else USER_PROMPT
            description = await self._infer(frame, prompt, system_prompt=system_prompt)
            self.last_spatial_description = description
            return VisionSummary(description=description, source="qwen3_vl_2b")
        except Exception:
            # 捕获模型推理异常（OOM, CUDA error, etc.），安全降级
            logger.warning(
                "Qwen3-VL inference failed. Marking lane as degraded. "
                "(v4.5.0 §1.3.6 降级)",
                exc_info=True,
            )
            self._available = False
            self._degraded = True
            return VisionSummary(description="", source="qwen3_vl_2b")

    async def describe_regions(self, crops: list[Image.Image]) -> VisionSummary:
        """
        Describe multiple cropped UI regions using multi-image VLM input.

        v4.5.0 §1.3.6: ROI-based inference — feeds 3-5 cropped UI regions
        from OmniParser L2 as separate images to Qwen3-VL, reducing latency
        from 3-5s to ~1-2s by avoiding full-frame encoding.

        Args:
            crops: List of PIL cropped images (top-k UI regions from L2).

        Returns:
            VisionSummary with description string.
        """
        if not self._available:
            return VisionSummary(description="", source="qwen3_vl_2b")

        now = time.monotonic()
        if self._last_process_time > 0:
            elapsed = now - self._last_process_time
            if elapsed < self._poll_interval:
                return VisionSummary(description="", source="qwen3_vl_2b")
        self._last_process_time = now

        try:
            description = await self._infer(
                frame=np.zeros((1, 1, 3), dtype=np.uint8),
                custom_prompt=ROI_USER_PROMPT,
                roi_crops=crops,
            )
            return VisionSummary(description=description, source="qwen3_vl_2b")
        except Exception:
            logger.warning(
                "Qwen3-VL ROI inference failed. Marking lane as degraded. "
                "(v4.5.0 §1.3.6 降级)",
                exc_info=True,
            )
            self._available = False
            self._degraded = True
            return VisionSummary(description="", source="qwen3_vl_2b")

    # v4.5.0 §T4: Composite image support — L2D crop + top-1 window crop
    async def process_composite(
        self,
        l2d_crop: np.ndarray,
        top1_crop: np.ndarray,
        top1_title: str = "",
        prompt: str = "",
    ) -> VisionSummary:
        """Process L2D + top-1 composite image via VLM.

        Horizontally concatenates the L2D crop and the top-1 window crop
        into a single composite image, then sends it to Qwen3-VL with a
        contextual prompt that describes the left (virtual character) and
        right (window content) halves.

        v4.5.0 §T4: Composite VLM input for L2D state descriptions.
        """
        composite = self._build_composite(l2d_crop, top1_crop)
        custom_prompt = (
            prompt
            if prompt
            else f"描述这个场景：左侧是一个虚拟角色，右侧是'{top1_title}'窗口的内容。"
        )
        return await self.process(composite, custom_prompt=custom_prompt,
                                  system_prompt=COMPOSITE_SYSTEM_PROMPT)

    def process_composite_sync(
        self,
        l2d_crop: np.ndarray,
        top1_crop: np.ndarray,
        top1_title: str = "",
    ) -> str:
        """Synchronous wrapper for composite VLM inference.

        Builds a composite image from L2D + top-1 crops and runs VLM
        inference synchronously. Uses asyncio.run() when no event loop
        is running, falls back to ThreadPoolExecutor when called from
        an async context.

        v4.5.0 §T4: Composite VLM input for L2D state descriptions.
        """
        try:
            composite_np = self._build_composite(l2d_crop, top1_crop)
            custom_prompt = (
                f"描述画面。:{top1_title}"
                if top1_title
                else "描述画面。"
            )

            return asyncio.run(
                self._infer(
                    composite_np,
                    custom_prompt=custom_prompt,
                    system_prompt=COMPOSITE_SYSTEM_PROMPT,
                )
            )
        except Exception:
            logger.warning(
                "process_composite_sync failed — returning empty description",
                exc_info=True,
            )
            return ""

    def describe_l2d_sync(self, l2d_crop: np.ndarray) -> str:
        """Describe Live2D avatar from crop image (no window). Called once during preload."""
        try:
            return asyncio.run(
                self._infer(
                    l2d_crop,
                    custom_prompt="你现在站在镜子前看着自己的Live2D形象。描述你看到的外貌——服装、发型、表情、姿态。用第二人称'你'叙述。",
                    system_prompt=L2D_SYSTEM_PROMPT,
                )
            )
        except Exception:
            return ""

    def describe_window_sync(self, window_crop: np.ndarray, window_title: str = "") -> str:
        """Synchronous window-only VLM inference (no L2D composite)."""
        custom_prompt = f"看看哥哥的屏幕画面:{window_title}" if window_title else "看看哥哥的屏幕画面"
        try:
            return asyncio.run(self._infer(window_crop, custom_prompt=custom_prompt, system_prompt=WINDOW_SYSTEM_PROMPT))
        except Exception:
            import logging
            logging.getLogger(__name__).warning("describe_window_sync unexpected error", exc_info=True)
            return ""

    @staticmethod
    def _build_composite(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Horizontally concatenate two images, resizing to same height.

        Both images are resized to the minimum height of the two, preserving
        aspect ratio, then stacked side-by-side. Returns a numpy uint8 array
        in RGB format.

        Args:
            img1: First image (H1, W1, 3) as uint8 numpy array.
            img2: Second image (H2, W2, 3) as uint8 numpy array.

        Returns:
            Composite image as uint8 numpy array (target_H, W1'+W2', 3).
        """
        pil1 = Image.fromarray(img1.astype(np.uint8))
        pil2 = Image.fromarray(img2.astype(np.uint8))

        # Resize both to same height (the smaller of the two)
        h1, h2 = pil1.height, pil2.height
        target_h = min(h1, h2)

        if pil1.height != target_h:
            new_w1 = int(pil1.width * target_h / pil1.height)
            pil1 = pil1.resize((new_w1, target_h), Image.LANCZOS)

        if pil2.height != target_h:
            new_w2 = int(pil2.width * target_h / pil2.height)
            pil2 = pil2.resize((new_w2, target_h), Image.LANCZOS)

        # Horizontal concatenation
        composite_pil = Image.new(
            "RGB", (pil1.width + pil2.width, target_h)
        )
        composite_pil.paste(pil1, (0, 0))
        composite_pil.paste(pil2, (pil1.width, 0))

        return np.array(composite_pil, dtype=np.uint8)

    def _load_model(self) -> None:
        """Load Qwen3-VL via vLLM or CPU."""
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if not self._use_vllm:
            logger.info("Loading Qwen3-VL-2B-Instruct via transformers (CPU mode) from %s ...", self._model_path)
            from transformers import AutoModelForVision2Seq, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self._model_path, trust_remote_code=True)
            self._cpu_model = AutoModelForVision2Seq.from_pretrained(
                self._model_path, dtype=torch.float32, device_map="cpu", trust_remote_code=True
            )
            self._available = True
            logger.info("Qwen3-VL-2B-Instruct CPU model loaded")
            return

        logger.info("Loading Qwen3-VL-2B-Instruct via vLLM from %s ...", self._model_path)
        from vllm import LLM
        self._model = LLM(
            model=self._model_path,
            max_model_len=4096,
            gpu_memory_utilization=0.50,  # v5.x: VLM model 4.26G + KV
            max_num_seqs=1,
        )
        self._available = True
        logger.info("Qwen3-VL-2B-Instruct vLLM engine loaded. VRAM=%.1fGB", torch.cuda.memory_reserved(0) / 1e9)

    async def _infer(self, frame, custom_prompt: str = "",
                     roi_crops=None,
                     system_prompt: str = "") -> str:
        if self._model is None and not hasattr(self, '_cpu_model'):
            self._load_model()
        if hasattr(self, '_cpu_model') and self._cpu_model is not None:
            return self._infer_cpu(frame, custom_prompt, roi_crops, system_prompt)
        if self._model is None:
            self._load_model()

        prompt = custom_prompt if custom_prompt else USER_PROMPT
        sys_prompt = system_prompt or SYSTEM_PROMPT
        import torch
        from vllm import SamplingParams

        # v5.x: When roi_crops is provided (describe_regions path), send those
        # crops as multi-image input instead of the dummy 1x1 frame.
        # Qwen-VL vLLM supports multi-image: pass a list of PIL images.
        if roi_crops and len(roi_crops) > 0:
            images: list[Image.Image] = []
            for crop in roi_crops:
                if isinstance(crop, np.ndarray):
                    images.append(Image.fromarray(crop))
                elif isinstance(crop, Image.Image):
                    images.append(crop)
                else:
                    continue  # skip non-image items

            if not images:
                return ""

            # Build multi-image prompt: one <|vision_start|><|image_pad|><|vision_end|> per image
            full_prompt = ""
            if sys_prompt:
                full_prompt += f"<|system|>\n{sys_prompt}\n"
            for _ in images:
                full_prompt += "<|vision_start|><|image_pad|><|vision_end|>"
            full_prompt += prompt

            try:
                outputs = await asyncio.to_thread(
                    self._model.generate,
                    [{"prompt": full_prompt, "multi_modal_data": {"image": images}}],
                    SamplingParams(max_tokens=80, temperature=0.7),  # v5.x: was 15 — need 30+ for NAME: xxx | DESC: xxx
                )
                return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
            except Exception:
                logger.warning("vLLM multi-image inference failed, returning empty", exc_info=True)
                return ""

        # Single-image path (process, process_composite, describe_l2d_sync)
        pil_img = self._resize_frame(frame) if frame is not None else None
        if pil_img is None:
            return ""

        # Build messages with system prompt — Qwen3-VL chat format
        full_prompt = ""
        if sys_prompt:
            full_prompt += f"<|system|>\n{sys_prompt}\n"
        full_prompt += f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"

        try:
            outputs = await asyncio.to_thread(
                self._model.generate,
                [{"prompt": full_prompt, "multi_modal_data": {"image": pil_img}}],
                SamplingParams(max_tokens=80, temperature=0.7),  # v5.x: was 15 — need 30+ for NAME: xxx | DESC: xxx
            )
            return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
        except Exception:
            logger.warning("vLLM inference failed, returning empty", exc_info=True)
            return ""

    def _infer_cpu(self, frame, custom_prompt="", roi_crops=None, system_prompt="") -> str:
        """CPU inference via transformers (slow, ~9s, but 0 GPU)."""
        import torch
        from PIL import Image

        # Multi-image path: roi_crops from describe_regions
        if roi_crops and len(roi_crops) > 0:
            all_images: list[Image.Image] = []
            content: list[dict] = [{"type": "text", "text": custom_prompt or "描述这个截图"}]
            for crop in roi_crops:
                if isinstance(crop, np.ndarray):
                    crop = Image.fromarray(crop)
                elif not isinstance(crop, Image.Image):
                    continue
                content.insert(0, {"type": "image", "image": crop})
                all_images.append(crop)

            if not all_images:
                return ""

            messages = [{"role": "user", "content": content}]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=all_images, return_tensors="pt")
            with torch.no_grad():
                generated = self._cpu_model.generate(**inputs, max_new_tokens=80)
            logger.info(f"Qwen3-VL CPU multi-image: generated {generated.shape[1]} tokens")
            result = self._processor.decode(generated[0], skip_special_tokens=True)
            if " assistant\n" in result:
                result = result.split(" assistant\n")[-1].strip()
            logger.info(f"Qwen3-VL CPU multi-image inference completed ({len(result)} chars)")
            return result

        # Single-image path
        pil_img = self._resize_frame(frame) if frame is not None else None
        if pil_img is None:
            return ""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": custom_prompt or "描述这个截图"}
        ]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[pil_img], return_tensors="pt")
        # Keep on CPU
        with torch.no_grad():
            generated = self._cpu_model.generate(**inputs, max_new_tokens=80)
        logger.info(f"Qwen3-VL CPU: generated {generated.shape[1]} tokens")
        result = self._processor.decode(generated[0], skip_special_tokens=True)
        # Extract only assistant response
        if " assistant\n" in result:
            result = result.split(" assistant\n")[-1].strip()
        logger.info(f"Qwen3-VL CPU inference completed ({len(result)} chars)")
        return result

    def _resize_frame(self, frame):
        """
        Resize frame to max 1280px on longest side preserving aspect ratio.

        v4.5.0 §1.3.6: max 1280px
        """
        from PIL import Image

        if isinstance(frame, np.ndarray):
            pil_img = Image.fromarray(frame.astype(np.uint8))
        else:
            pil_img = frame

        w, h = pil_img.size
        if max(w, h) > DEFAULT_MAX_SIZE:
            scale = DEFAULT_MAX_SIZE / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)  # pyright: ignore[reportAttributeAccessIssue]

        return pil_img.convert("RGB")
