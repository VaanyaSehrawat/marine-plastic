#!/usr/bin/env bash
# Step 4: record D-09 and the status, commit and push. Does NOT start the dashboard.
# Place in ~/Projects/marine-plastic/scripts/ and run: bash scripts/integrate_step4.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
[ -f build/step3.ok ] || { echo "STEP 4 BLOCKED: Step 3 has not passed (build/step3.ok missing)"; exit 1; }

echo "[1/4] recording decisions"
if grep -q "D-09" DECISIONS.md; then
  echo "  D-09 already in DECISIONS.md (kept)"
else
  cat >> DECISIONS.md <<'DEOF'

## Update 15 Nov 2026 - D-09 (detector and learned tile scoring)
| ID | Decision | Reason / evidence |
|---|---|---|
| D-09 | Dashboard default detector is YOLO11m (YOLO11s stays available with `MODEL_DIR=yolo11s_1280 bash scripts/run_dashboard.sh`). Learned tile priority uses a 71,233-parameter route-context transformer (ONNX, run in Java), trained on val-split detections, selectable in the dashboard next to the heuristic scoring. | Detectors (official splits, 1280 px, conf 0.001, NMS IoU 0.6, one training run each): YOLO11m val mAP50-95 0.182 and test 0.157; YOLO11s val 0.162 and test 0.144. Java CPU inference on the student's Mac (3 fixture images, after one warm-up): YOLO11m 1835-1879 ms, YOLO11s 549-694 ms. Tile ranking on test (610 images with at least 5 ground-truth tiles): Spearman transformer 0.661 +- 0.001, tile MLP 0.660 +- 0.001, detector-only score 0.531, random -0.007; top-5 overlap 0.591, 0.590, 0.524, 0.095. Cross-tile context gave no measurable gain over the MLP. Labels are ground-truth boxes weighted by the same class weights, so this measures agreement with a proxy, not cleanup efficiency. Java checks: scripts/test_route.sh and scripts/run_dashboard.sh --selftest PASS. |
DEOF
  echo "  appended D-09 to DECISIONS.md"
fi
if grep -q "Status 15 Nov 2026 (integration)" HANDOFF.md; then
  echo "  status block already in HANDOFF.md (kept)"
else
  cat >> HANDOFF.md <<'DEOF'

## Status 15 Nov 2026 (integration)
- M4: YOLO11m selected (val mAP50-95 0.182 vs 0.162); Java fixture test PASS (scripts/test_route.sh).
- M7: route-context transformer integrated in Java (RouteScorer, RouteTest); dashboard has a Scoring selector. Run: bash scripts/run_dashboard.sh (self-test: --selftest). YOLO11s heuristic path: bash scripts/run_app.sh.
- VAE and DCGAN are offline Python results (results/generative, docs/RESULTS.md); no Java inference needed.
- Pending: evaluation package (YOLO11m error analysis, planner vs serpentine over the test split), report, demo script.
DEOF
  echo "  appended status block to HANDOFF.md"
fi

echo "[2/4] staging"
git add java/app/src scripts DECISIONS.md HANDOFF.md
git add -f models/yolo11m_1280/model_info.json models/yolo11m_1280/SHA256.txt models/yolo11m_1280/fixture \
           models/route/route_model.onnx models/route/route_model_info.json models/route/route_fixture.json models/route/route_data.json.gz 2>&1 | sed 's/^/  /'
echo "== safety check on staged files =="
BAD=0
for f in $(git diff --cached --name-only); do
  case "$f" in
    *.pt|*.pth|*.zip|*.tar.gz) echo "  FORBIDDEN staged file: $f"; BAD=1;;
    *.onnx) [ "$f" = "models/route/route_model.onnx" ] || { echo "  FORBIDDEN staged ONNX: $f"; BAD=1; };;
  esac
  if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 5000000 ]; then echo "  FILE OVER 5 MB staged: $f"; BAD=1; fi
done
if [ $BAD -ne 0 ]; then git reset -q; echo "STEP 4 FAILED: unstaged everything, nothing committed. Send me this output."; exit 1; fi
echo "  staged files: $(git diff --cached --name-only | wc -l | tr -d ' ') | largest staged file: $(for f in $(git diff --cached --name-only); do [ -f "$f" ] && echo "$(wc -c < "$f") $f"; done | sort -rn | head -1)"

echo "[3/4] commit"
if git diff --cached --quiet; then
  echo "  nothing new to commit"
else
  git commit -q -m "feat(m7): Java route-context transformer (ONNX), YOLO11m detector, learned scoring in dashboard, tests, D-09" || { echo "STEP 4 FAILED: commit failed"; exit 1; }
  echo "  committed: $(git log --oneline | head -1)"
fi

echo "[4/4] push"
git push > build/_push4.txt 2>&1; prc=$?
tail -3 build/_push4.txt | sed 's/^/  /'
if [ $prc -ne 0 ]; then echo "STEP 4: COMMITTED LOCALLY, PUSH FAILED (exit $prc). Send me the lines above."; exit 1; fi
echo "  $(git status -sb | head -1)"
echo "  tracked detector weights (must be empty): [$(git ls-files models | grep -E 'yolo.*\.(pt|onnx)$')]"
touch build/step4.ok
echo; echo "STEP 4: DONE"
