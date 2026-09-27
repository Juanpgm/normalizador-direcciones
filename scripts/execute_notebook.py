"""Execute direcciones_ANN.ipynb in place with nbclient and verify the outputs."""

from __future__ import annotations

import os
import sys
import time

import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "direcciones_ANN.ipynb")

nb = nbf.read(NB, as_version=4)
client = NotebookClient(
    nb,
    timeout=-1,
    kernel_name="python3",
    resources={"metadata": {"path": ROOT}},
    allow_errors=True,
    store_widget_state=True,
)
t0 = time.time()
client.execute()
nbf.write(nb, NB)
print(f"executed in {time.time() - t0:.0f}s", flush=True)

errors = []
empty = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    outs = cell.get("outputs", [])
    for o in outs:
        if o.get("output_type") == "error":
            errors.append((i, o.get("ename"), " | ".join(o.get("traceback", []))[-1600:]))
    if not outs and cell.source.strip():
        empty.append(i)

print(f"code cells: {sum(1 for c in nb.cells if c.cell_type == 'code')}")
print(f"cells with NO output: {empty}")
print(f"cells with an error : {[e[0] for e in errors]}")
for i, name, tb in errors:
    print("=" * 100)
    print(f"cell {i}: {name}")
    print(nb.cells[i].source[:600])
    print("-" * 60)
    print(tb)
sys.exit(1 if errors else 0)
