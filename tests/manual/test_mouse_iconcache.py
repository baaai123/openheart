#!/usr/bin/env python3
"""模拟 LLM 命令 → IconCache 查询 → 鼠标移动测试"""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

async def main():
    from src.perception.visual.screenshot import capture_screenshot
    from src.perception.visual.visual_pipeline import VisualPipeline
    from src.perception.visual.icon_cache import get_icon_cache
    from src.execution.channels.mouse_channel import MouseChannel
    from src.config.runtime import RuntimeConfig

    cfg = RuntimeConfig.from_environ()
    print("1. 运行视觉管道填充 IconCache...")
    pipeline = VisualPipeline(cfg)
    _ = pipeline.process_frame_sync(capture_screenshot(), skip_vlm=True)
    snapshot = pipeline.process_frame_sync(capture_screenshot(), skip_vlm=True)

    cache = get_icon_cache()
    total = sum(1 for _ in cache._cache.values())
    print(f"   IconCache: {total} 图标已缓存")

    # 显示几个缓存条目
    for name, data in list(cache._cache.items())[:5]:
        print(f"   {name} → ({data['coord'][0]:.0f},{data['coord'][1]:.0f}) conf={data['conf']:.2f}")

    # 模拟 LLM 命令: {{move:回收站}}
    targets = ["回收站", "此电脑", "控制面板", "百度网盘", "Google Chrome", "终端"]
    found = []
    mouse = MouseChannel()
    for t in targets:
        result = cache.query(t)
        if result:
            x, y = result[:2]
            tier = result[2] if len(result) > 2 else "?"
            print(f"\n2. 查询: \"{t}\" → ({x:.0f},{y:.0f}) [{tier}]")
            found.append((t, int(x), int(y)))
            print(f"   Bezier 移动到目标 ({int(x)},{int(y)})...")
            await mouse.move_to(int(x), int(y))
            print(f"   ✅ Bezier 移动完成 → ({int(x)},{int(y)})")

    if not found:
        print("\n   ❌ 所有目标均未命中 IconCache")
        print("   (尝试说一句包含回收站/此电脑的对话触发 LLM 命令)")
        return

    print("\n✅ 完整链路测试通过")

asyncio.run(main())
