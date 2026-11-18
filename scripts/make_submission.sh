#!/usr/bin/env bash
# Builds a submission folder on the Desktop. Place in ~/Projects/marine-plastic/scripts/ and run:
#   bash scripts/make_submission.sh
# Read-only for the repository: it commits nothing and changes no project file.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
ROOT="$(pwd)"
STAMP="$(date +%Y%m%d)"
OUT="$HOME/Desktop/marine-plastic-submission-$STAMP"

echo "[1/6] checking the repository state"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "STOP: uncommitted changes in tracked files:"; git status --short --untracked-files=no | head -10
  echo "Commit or discard them first, then rerun."; exit 1
fi
git fetch -q 2>/dev/null
AHEAD="$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)"
[ "$AHEAD" = "0" ] || { echo "STOP: $AHEAD commit(s) not pushed. Run: git push"; exit 1; }
grep -q "Failed: 0" docs/VERIFICATION.md 2>/dev/null || { echo "STOP: docs/VERIFICATION.md missing or shows failures. Run: bash scripts/verify_all.sh"; exit 1; }
VER_HASH="$(git log -1 --format=%h -- docs/VERIFICATION.md)"
echo "  clean and pushed | HEAD $(git rev-parse --short HEAD) | last verification commit $VER_HASH"
if [ "$(git rev-list --count "$VER_HASH"..HEAD -- java scripts models)" != "0" ]; then
  echo "  WARNING: code, scripts or models changed after the last verification run. Rerun scripts/verify_all.sh before submitting."
fi

echo "[2/6] creating $OUT"
[ -e "$OUT" ] && { echo "STOP: $OUT already exists. Move or delete it, then rerun."; exit 1; }
mkdir -p "$OUT/code" "$OUT/weights" "$OUT/documents" "$OUT/figures"

echo "[3/6] source archive (tracked files only)"
git archive --format=zip --prefix=marine-plastic/ -o "$OUT/code/marine-plastic-source.zip" HEAD || { echo "FAILED: git archive"; exit 1; }
echo "  $(du -h "$OUT/code/marine-plastic-source.zip" | cut -f1) | $(unzip -l "$OUT/code/marine-plastic-source.zip" | tail -1 | awk '{print $2}') files"

echo "[4/6] weights (not in git) with SHA-256 files"
MISS=0
for f in models/yolo11m_1280/yolo11m_1280_best.onnx models/yolo11m_1280/yolo11m_1280_best.pt \
         models/yolo11s_1280/yolo11s_1280_best.onnx models/yolo11s_1280/yolo11s_1280_best.pt \
         models/route/route_model.onnx results/generative/weights/vae.pt results/generative/weights/gan_generator.pt; do
  if [ -s "$f" ]; then cp "$f" "$OUT/weights/"; else echo "  MISSING $f"; MISS=1; fi
done
( cd "$OUT/weights" && shasum -a 256 * > SHA256.txt )
echo "  $(ls "$OUT/weights" | grep -vc SHA256) weight files, $(du -sh "$OUT/weights" | cut -f1)"
[ $MISS -eq 0 ] || echo "  WARNING: some weights were missing (listed above)"

echo "[5/6] documents and figures"
cp README.md "$OUT/documents/" 2>/dev/null
for f in docs/*.md; do cp "$f" "$OUT/documents/"; done
cp results/detectors/plots/*.png "$OUT/figures/" 2>/dev/null
cp results/generative/vae_recon.png results/generative/gan_samples.png "$OUT/figures/" 2>/dev/null
DEMO=""; for c in "$HOME/Desktop/demo_recording.mp4" "$HOME/Movies/demo_recording.mp4" "$HOME/Downloads/demo_recording.mp4"; do [ -f "$c" ] && { DEMO="$c"; break; }; done
if [ -n "$DEMO" ]; then mkdir -p "$OUT/demo"; cp "$DEMO" "$OUT/demo/"; echo "  demo video copied: $DEMO"; else echo "  NOTE: demo_recording.mp4 not found on Desktop, Movies or Downloads (add it to $OUT/demo/ yourself)"; fi
echo "  $(ls "$OUT/documents" | wc -l | tr -d ' ') documents, $(ls "$OUT/figures" | wc -l | tr -d ' ') figures"

echo "[6/6] manifest and restore check"
{
  echo "Marine Plastic Detection and Coastal Cleanup Planner - submission package"
  echo "Built: $(date '+%Y-%m-%d %H:%M') | git commit $(git rev-parse HEAD)"
  echo "Repository: $(git remote get-url origin 2>/dev/null)"
  echo
  echo "code/marine-plastic-source.zip  tracked source (Java, Python, notebooks, scripts, docs, results, small models)"
  echo "weights/                        detector, route, VAE and GAN weights with SHA-256 (not stored in git)"
  echo "documents/                      README, report draft, results, evaluations, demo script, jury Q&A, verification"
  echo "figures/                        confusion matrices, PR and F1 curves, training curves, VAE and GAN figures"
  echo
  echo "SHA-256 of every file:"
  ( cd "$OUT" && find . -type f ! -name SUBMISSION_MANIFEST.txt | sort | xargs shasum -a 256 )
} > "$OUT/SUBMISSION_MANIFEST.txt"
TMP="$(mktemp -d)"; unzip -q "$OUT/code/marine-plastic-source.zip" -d "$TMP"
RC=0
for f in marine-plastic/README.md marine-plastic/setup.sh marine-plastic/scripts/verify_all.sh marine-plastic/scripts/run_dashboard.sh \
         marine-plastic/java/app/src/main/java/mp/app/detect/YoloOnnxDetector.java marine-plastic/java/fundamentals/src/main/java/mp/fund/Svd.java \
         marine-plastic/docs/REPORT.md marine-plastic/docs/VERIFICATION.md marine-plastic/models/route/route_model.onnx marine-plastic/notebooks/INDEX.md; do
  [ -s "$TMP/$f" ] || { echo "  RESTORE CHECK: missing $f in the archive"; RC=1; }
done
rm -rf "$TMP"
[ $RC -eq 0 ] && echo "  restore check: key files present in the archive"
echo
echo "Package: $OUT ($(du -sh "$OUT" | cut -f1))"
( cd "$OUT" && find . -maxdepth 2 | sort | head -40 )
[ $RC -eq 0 ] && echo && echo "SUBMISSION PACKAGE: BUILT" || { echo; echo "SUBMISSION PACKAGE: BUILT WITH PROBLEMS (see above)"; exit 1; }
