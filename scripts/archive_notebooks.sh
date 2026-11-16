#!/usr/bin/env bash
# Archive the notebooks that produced the models and results (Kaggle training, ONNX export, route data, transformer, VAE/GAN; Colab dataset review).
# Before running: in Kaggle use File > Download notebook (.ipynb); in Colab use File > Download > Download .ipynb.
# Place this file in ~/Projects/marine-plastic/scripts/ and run: bash scripts/archive_notebooks.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p notebooks build logs scripts
PY=".venv/bin/python"; [ -x "$PY" ] || PY=".tools/pybin/python3.12"; [ -x "$PY" ] || PY="python3"

cat > scripts/archive_notebooks.py <<'PYEOF'
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
PYEOF
"$PY" -m py_compile scripts/archive_notebooks.py || { echo "FAILED: script syntax error"; exit 1; }
"$PY" scripts/archive_notebooks.py; rc=$?
[ $rc -eq 0 ] || { echo "ARCHIVE FAILED (exit $rc): nothing committed."; exit 1; }

echo "== commit and push =="
git add notebooks scripts/archive_notebooks.py scripts/archive_notebooks.sh
BAD=0
for f in $(git diff --cached --name-only); do
  case "$f" in *.pt|*.pth|*.onnx|*.zip|*.tar.gz) echo "  FORBIDDEN staged file: $f"; BAD=1;; esac
  if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 3000000 ]; then echo "  FILE OVER 3 MB staged: $f"; BAD=1; fi
done
if [ $BAD -ne 0 ]; then git reset -q; echo "FAILED: unstaged everything, nothing committed."; exit 1; fi
if git diff --cached --quiet; then
  echo "  nothing new to commit"
else
  git commit -q -m "docs: archive the Kaggle and Colab notebooks (outputs stripped) that produced the models and results" || { echo "FAILED: commit failed"; exit 1; }
  echo "  committed: $(git log --oneline | head -1)"
fi
git push > build/_push_nb.txt 2>&1; prc=$?
tail -3 build/_push_nb.txt | sed 's/^/  /'
[ $prc -eq 0 ] || { echo "COMMITTED LOCALLY, PUSH FAILED (exit $prc)"; exit 1; }
echo "  $(git status -sb | head -1)"
echo; echo "NOTEBOOKS: COMMITTED AND PUSHED"
