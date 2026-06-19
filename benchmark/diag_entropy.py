"""Chern熵值分布诊断——帮助选择合适的阈值"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chern_engine import ChernEngine

DIRS = [(0,-1),(1,0),(0,1),(-1,0)]

def gen_map(size, density, seed=42):
    rng = random.Random(seed)
    grid = [[0]*size for _ in range(size)]
    n = int(size*size*density)
    cs = [(x,y) for x in range(size) for y in range(size) if not ((x==0 and y==0) or (x==size-1 and y==size-1))]
    rng.shuffle(cs)
    for x,y in cs[:n]: grid[y][x]=1
    return grid

for size in [20, 40, 60]:
    for density in [0.1, 0.2, 0.3]:
        grid = gen_map(size, density, seed=42)
        engine = ChernEngine(entropy_threshold=0.0)  # 返回原始熵值

        entropies = []
        # 随机采样200个空地
        empty = [(x,y) for x in range(size) for y in range(size) if not grid[y][x]]
        rng = random.Random(99)
        pts = rng.sample(empty, min(200, len(empty)))

        for x, y in pts:
            e, probs = engine._qw_entropy((x, y), size)
            entropies.append(e)

        entropies.sort()
        p10, p50, p70, p80, p90 = (
            entropies[int(len(entropies)*0.1)],
            entropies[int(len(entropies)*0.5)],
            entropies[int(len(entropies)*0.7)],
            entropies[int(len(entropies)*0.8)],
            entropies[int(len(entropies)*0.9)],
        )
        high_count = sum(1 for e in entropies if e >= 1.2)
        print(f"size={size:2d} den={density:.1f} | P10={p10:.2f} P50={p50:.2f} P70={p70:.2f} P80={p80:.2f} P90={p90:.2f} | >=1.2: {high_count}/{len(entropies)} ({high_count/len(entropies)*100:.0f}%)")
