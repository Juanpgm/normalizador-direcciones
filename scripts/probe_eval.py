import os
import pandas as pd
from unidecode import unidecode

CTX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "context")
SPECS = [
    ("20260821 BD Inmuebles asegurados Cali - Fasecolda.xlsx", "EXPUESTOS", 0),
    ("RUD VALLE DEL CAUCA - CALI.xlsx", "RUD MUNICIPIO", 0),
    ("inspecciones_2026-09-23_17-27.xlsx", "inspecciones", 2),
    ("stickers_2026-09-23_20-23.xlsx", "stickers", 5),
    ("acciones_candidatos_demolicion_2026-09-23_20-25.xlsx", "acciones", 2),
]
for fn, sheet, hdr in SPECS:
    df = pd.read_excel(os.path.join(CTX, fn), sheet_name=sheet, header=hdr, nrows=8)
    print("=" * 90)
    print(fn, "| rows(sample)", len(df))
    for c in df.columns:
        print("   ", repr(c), "->", unidecode(str(c)).strip().lower())
