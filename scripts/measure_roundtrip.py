import os
import sys
import random

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cali_address.parser import parse_address, canonical  # noqa: E402

df = pd.read_parquet(os.path.join(ROOT, "artifacts", "catastro.parquet"))
d = df["direccion"].dropna().astype(str)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
sample = d.sample(N, random_state=0).tolist()

LEGACY = {"K", "C", "A", "D", "T", "B", "P", "U", "V", "L", "S", "E", "Z", "M", "J"}

ok = 0
fails = []
ok_modern = 0
n_modern = 0
for s in sample:
    p = parse_address(s)
    c = canonical(p, style="spaced", with_complement=True)
    norm_target = " ".join(s.split())
    modern = s.split()[0] not in LEGACY
    n_modern += modern
    if c == norm_target:
        ok += 1
        ok_modern += modern
    elif len(fails) < 4000:
        fails.append((s, c, p.notes))
print(f"sample={N} roundtrip_exact={ok/N:.4f}  modern_prefix_only={ok_modern/max(n_modern,1):.4f} (n={n_modern})")
random.seed(0)
for s, c, notes in random.sample(fails, min(45, len(fails))):
    print(f"  IN  {s!r}\n  OUT {c!r}  {notes}")
