import json, pathlib, re, sys, time

dl = pathlib.Path.home() / "Downloads"
cut = time.time() - 10 * 86400
paths = sorted([p for p in dl.glob("*.ipynb") if p.stat().st_mtime >= cut], key=lambda p: p.stat().st_mtime)
if not paths:
    sys.exit("NO NOTEBOOKS: no .ipynb file modified in the last 10 days was found in ~/Downloads")
MARK = {
    "dataset restore and YOLO conversion": "bepli.yaml",
    "YOLO11s training": "yolo11s.pt",
    "YOLO11m training": "yolo11m.pt",
    "ONNX export": "format=\"onnx\"",
    "letterbox parity / detection dump": "def letterbox",
    "route transformer": "class RouteTransformer",
    "VAE": "class VAE",
    "DCGAN": "class G(nn.Module)",
    "Setup cell (checkpoint restore)": "SETUP OK",
    "Colab dataset review": "LEAKAGE AUDIT",
}
found = {k: [] for k in MARK}
out_dir = pathlib.Path("notebooks"); out_dir.mkdir(exist_ok=True)
for p in paths:
    try:
        nb = json.load(open(p))
    except Exception as e:
        print(f"skipped {p.name}: not valid JSON ({str(e)[:60]})")
        continue
    code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    text = "\n".join("".join(c.get("source", [])) for c in code)
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            c["outputs"] = []; c["execution_count"] = None
        c["metadata"] = {}
    nb["metadata"] = {k: v for k, v in nb.get("metadata", {}).items() if k in ("kernelspec", "language_info")}
    slug = re.sub(r"[^a-z0-9]+", "-", p.stem.lower()).strip("-") or "notebook"
    dst = out_dir / f"{slug}.ipynb"
    json.dump(nb, open(dst, "w"), indent=1)
    kb = dst.stat().st_size / 1024
    print(f"\n{p.name} -> {dst} | {len(nb.get('cells', []))} cells ({len(code)} code) | {kb:.0f} KB after stripping outputs"
          + ("  WARNING: over 2 MB" if kb > 2048 else ""))
    for i, c in enumerate(code, 1):
        lines = [l.strip() for l in "".join(c.get("source", [])).split("\n") if l.strip() and not l.strip().startswith(("import ", "from ", "#", "%"))]
        print(f"   code cell {i}: {(lines[0] if lines else '(empty)')[:80]}")
    for k, m in MARK.items():
        if m in text:
            found[k].append(dst.name)
print("\n== pipeline stages found in the archived notebooks ==")
for k, v in found.items():
    print(f"  {'FOUND  ' if v else 'MISSING'} {k}" + (f"  ({', '.join(sorted(set(v)))})" if v else ""))
print("ARCHIVE: DONE")
