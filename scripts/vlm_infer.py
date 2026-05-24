#!/usr/bin/env python3
"""MiniCPM-V-4.6 inference — subprocess from PromptLearner."""
import sys, os
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
MODEL_PATH = "models/minicpm-v-4.6/OpenBMB/MiniCPM-V-4___6"
def main():
    if len(sys.argv) < 2: sys.exit(1)
    crop_path = sys.argv[1]
    if not os.path.exists(crop_path): sys.exit(1)
    if not hasattr(main, "model"):
        main.processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        main.model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH, trust_remote_code=True, dtype="float32")
    image = Image.open(crop_path).convert("RGB")
    prompt = "这是UI裁剪区域。用格式: NAME: snake_case | DESC: 中文描述"
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = main.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = main.processor(text=[text], images=[image], return_tensors="pt")
    import torch
    with torch.no_grad(): generated = main.model.generate(**inputs, max_new_tokens=60)
    result = main.processor.decode(generated[0], skip_special_tokens=True)
    if "assistant" in result: result = result.split("assistant")[-1].strip()
    print(result)
if __name__ == "__main__": main()
