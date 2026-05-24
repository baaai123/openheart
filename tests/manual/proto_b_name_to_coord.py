#!/usr/bin/env python3
"""Proto B: Name-to-coordinate matching for IconCache query logic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

CACHE = {
    "回收站": (120, 700), "此电脑": (120, 200), "百度网盘": (106, 100),
    "控制面板": (120, 800), "Google Chrome": (200, 300),
    "Visual Studio Code": (300, 200), "终端": (500, 500),
}

def char_overlap(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0
    return len(sa & sb) / len(sa | sb)

def query(name, cache=None, mouse_pos=None):
    if cache is None: cache = CACHE
    if name in cache: return cache[name], 'exact'
    for k, v in cache.items():
        if name.lower() in k.lower() or k.lower() in name.lower(): return v, 'substring'
    best, best_v, best_s = 0, None, 0
    for k, v in cache.items():
        s = char_overlap(name, k)
        if s > best_s: best, best_v, best_s = k, v, s
    if best_s > 0.5: return best_v, f'overlap({best_s:.2f})'
    return None, 'miss'

def main():
    tests = ["回收站", "回收", "那个", "此电脑", "控制面板", "chrome", "notexist"]
    hits = 0
    for inp in tests:
        coord, tier = query(inp)
        hit = 'OK' if coord else 'XX'
        if coord: hits += 1
        print(f'{hit} "{inp}" -> {coord} ({tier})')
    print(f'Hits: {hits}/{len(tests)}')

if __name__ == "__main__":
    main()
