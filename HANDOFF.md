# HANDOFF.md - shared state (Claude = technical lead, ChatGPT = tutor/execution guide, student = owner)
Last updated: 2026-11-05 | Deadline **2026-11-22** (aim: submit 21 Nov) | Solo, 5-6 h/day, free tools only.
Read also: `DECISIONS.md` (what is approved/provisional/open), `docs/MILESTONES.md` (pass/fail), `docs/RESEARCH_NOTES.md` (all research, no need to redo), `docs/M3_DATASET_VALIDATION.md`.

## Roles
Claude: architecture, substantial code, milestone reviews, this file. ChatGPT: guides command execution, explains concepts, troubleshoots small errors on its own, reviews code, viva prep. Student: runs commands, reports real output verbatim, must be able to defend everything.
ChatGPT needs to involve Claude only for changes to architecture, dependency versions, dataset choice or scope; otherwise use `scripts/collect_diagnostics.sh` output and fix within the approved design.

## Approved architecture (do not change without a written reason)
Python/PyTorch (Colab) -> ONNX -> Java 21 app with ONNX Runtime Java 1.20.0 -> Java-served dashboard; pure-Java fundamentals module; image-space tile-grid planner (no GPS); torchvision Faster R-CNN + RetinaNet; fallback if Java inference fails: local Python HTTP detector behind a `Detector` interface.

## Student environment (reported)
MacBook M4 arm64, macOS 26.2 (25C56), system OpenJDK 17.0.19 + Python 3.13.6 (never modified), Git 2.39.5, ~15 GB free, VS Code, Maven NOT installed (not needed). Project root `~/Projects/marine-plastic` (was NOT a git repo: `git init` happens in setup).

## Pins (config/toolchain.env, requirements/*.txt)
Temurin 21.0.13+9 | uv 0.13.7 | CPython 3.12 (uv-managed) | **onnxruntime 1.20.0 (Python and Java jar)**, jar SHA-256 `c46608681692b3693914defb42fa3119ad7cb6146870581ece247e0ed5793fe8` (computed on the student's Mac) | onnx 1.17.0, numpy 2.1.3, Pillow 11.0.0, PyYAML 6.0.2, tqdm 4.67.1, pytest 8.3.4 | train extras: torch 2.6.0, torchvision 0.21.0, scipy 1.14.1, matplotlib 3.9.2, pandas 2.2.3, pycocotools 2.0.8, imagehash 4.3.1. Transitive deps float; lock them after the first good Mac install (`logs/env_report.txt` has pip freeze).

## Evidence legend: G generated | S sandbox (Linux x86_64) | M student's Mac | D real data

## Status
| Item | Evidence | Notes |
|---|---|---|
| setup.sh / env.sh / lib.sh | G, S | Linux dev mode (`MP_DEV_ALLOW_LINUX=1`): fresh install, idempotent re-run, `--check`, corrupt cache, broken JDK folder, checksum-mismatch stop + recovery. Student ran `--check` on the Mac (preflight passed). macOS-only branches (sw_vers, Rosetta, xcode-select, arm64 `file`) never executed in full on the Mac. |
| JDK 21 + uv downloads, pins | verified real macOS arm64 artifacts (SHA-256 recomputed from the actual download; layout `Contents/Home`; Mach-O ARM64) | not executed on Mac |
| Python pins | resolves for macOS arm64 / CPython 3.12 (`uv pip compile`); base set installed + tested on Linux | torch macOS binaries never executed; torch export path untested |
| ORT 1.20.0 Java jar | **M**: download OK, both osx-aarch64 dylibs inside, SHA-256 recorded | `fetch_java_deps.sh` enforces SHA-256 + both dylibs (strict check tested in S with fake jars: pass, wrong hash, missing dylib) |
| Python export + parity (`ml/smoke`) | S | onnx.helper builder: max diff vs independent NumPy reference 1.2e-07 |
| Java smoke test (`OnnxSmokeTest.java`) | S with REAL ORT 1.20.0 API (compiled from official v1.20.0 source) and real engine (libonnxruntime 1.20.0 from the Python wheel via a JNI bridge I built from the official C sources) on Linux: PASS, diff 1.192e-07 / 5.960e-08, identical to Python | **Not the official jar, not a Mac.** Mac run still required. No stub exists in the repo (guarded by `tests/python/test_repo_guards.py`). |
| M3 dataset tools (`ml/data/*`, notebook) | G, S on synthetic data | 28 tests pass (1 skip = torch). Never run on BePLi. |
| Everything else | not started | Detector, Java fundamentals, app, planner, dashboard, evaluation, docs, demo |

## Test evidence (this session, sandbox)
`pytest tests/python`: 28 passed, 1 skipped (with and without pycocotools installed). shellcheck: only false-positive "unused variable" warnings (variables are used by sourcing scripts). Real-ORT Linux smoke test: PASS (above). Fetch-script scenarios: good jar OK; rerun idempotent; wrong pin -> jar moved aside, new download rejected; jar with only `osx-arm64` -> FAIL with folder listing. Full dev-mode setup regression after the final edits: see the session report.
Static inspection only: macOS branches of setup.sh, `env.sh` zsh/bash behaviour was tested on Linux zsh/bash only, notebook code cells (JSON valid, syntax checked, never executed).

## Known issues / unverified
- Mac smoke test not run. Expected: see README. Failure modes and responses are in README troubleshooting.
- Dataset statistics are UNVERIFIED claims until M3 (see DATASET_CARD). BePLi archive layout and download URL unverified.
- `setup.sh` progress output from `curl` is noisy in logs (cosmetic).
- Git: the folder was not a repo; setup initialises it and commits only if `user.name`/`user.email` exist.

## Next tasks (in order)
1. **Student:** run the Mac sequence in README (extract, `bash setup.sh --minimal`, `source env.sh && bash scripts/run_tests.sh`). Report: full output of both commands, or `logs/diagnostics-*.txt` on failure. Pass = `ONNX smoke test PASSED` with `os.arch=aarch64` and `SETUP COMPLETE`.
2. **If M2 passes:** update this file's status (Mac rows -> M), run `bash setup.sh` (full) once, then start M3 on Colab per `docs/M3_DATASET_VALIDATION.md`.
3. **If the Java step fails** with natives/arch errors: do not reopen the version decision first; send diagnostics to Claude; fallback D-07 (HTTP detector) is pre-approved.
4. **Claude after M3:** builds the M4 training/eval/ONNX-export pipeline against the real measured classes (details at the end of docs/M3_DATASET_VALIDATION.md).
