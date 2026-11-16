# Demo script (about 7 minutes)

## Before the demo (do the day before and again 30 minutes before)

1. Plug in the laptop, close other heavy applications.
2. `cd ~/Projects/marine-plastic && bash scripts/verify_all.sh` and wait for `VERIFY: ALL CHECKS PASSED`.
3. Keep these tabs ready: the dashboard (http://127.0.0.1:8080), `docs/RESULTS.md`, `docs/DETECTOR_ANALYSIS.md`, `results/detectors/plots/yolo11m_test_confusion_matrix_normalized.png`, `results/generative/gan_samples.png`.
4. Have the recorded demo video and a copy of the repository on a USB stick.

## Live demo

| time | action | what to say (in your own words) |
|---|---|---|
| 0:00 | Start: `bash scripts/run_dashboard.sh` | The models were trained in Python on a free Kaggle GPU, exported to ONNX, and run here in Java with ONNX Runtime. |
| 0:45 | Choose `003278.png`, Scoring = Heuristic, Analyze | The detector finds litter boxes; each tile gets a score from class weights times confidence. The yellow route starts at an assumed entry tile in the top-left corner. |
| 2:00 | Switch Scoring to Learned (transformer), Analyze | A small transformer reads all tiles together and predicts a priority for each one. Different tiles get picked. |
| 3:00 | Move the Confidence and Tile threshold sliders | Lower confidence finds more objects but adds false ones. |
| 3:45 | Show the 7x9 grid with Learned selected | The transformer was trained for the 6x8 grid, so the dashboard falls back to the heuristic and says so in the panel. |
| 4:30 | Open `docs/PLANNER_EVAL.md` | Over 610 test images, learned scoring captures 0.785 of the best possible priority in the top 5 tiles, heuristic 0.717, random 0.194. The optimised route is shorter than a sweep on 88% of images. Priority is built from ground-truth boxes and assumed class weights, so this is agreement with a proxy. |
| 5:30 | Open `docs/DETECTOR_ANALYSIS.md` | About 90% of objects are tiny. Half of all objects get no detection at 0.25. Some classes are found but mislabelled. |
| 6:15 | Run `bash scripts/run_fundamentals.sh` or show its output | The Java fundamentals module has its own tensors, SVD, MLP and optimizers, with 100 automated checks. |
| 6:45 | Close with the limitations slide | Low absolute accuracy, one training run, proxy target, no out-of-domain test. |

Optional: upload your own beach photo (PNG or JPEG) through the dashboard. Say beforehand that performance on photographs outside the training beaches has not been measured.

## If something fails

- Dashboard slow or crashes: run `MODEL_DIR=yolo11s_1280 bash scripts/run_dashboard.sh` (about 3 times faster).
- Port 8080 busy: `lsof -ti tcp:8080 | xargs kill`, then start again.
- Java will not start: play the recorded video and show `docs/VERIFICATION.md`.

## Recorded demo

1. Press Cmd+Shift+5 and choose Record Entire Screen with the microphone on.
2. Follow the same table above, speaking slowly; keep it under 8 minutes.
3. Save as `demo_recording.mp4`. Do not commit it to git (large); keep it with the submission files.
