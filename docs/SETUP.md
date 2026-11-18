# Marine Plastic Detection and Coastal Cleanup Planner

> **Historical status line:** the status text below dates from 5 Nov 2026. Milestones 1 and 2 were later verified on the Mac; see docs/MILESTONES.md.


Individual B.Tech CSE (AI & ML) Deep Learning project, Ocean Conservation domain. **Deadline: 22 November 2026.**

**Architecture:** Python + PyTorch train/evaluate models on Colab -> ONNX -> a Java 21 application runs them with ONNX Runtime
Java, plans tile-based cleanup priorities in *image space* (never GPS) and serves a local dashboard. The neural-network
fundamentals module (tensors, SVD, MLP, optimizers, learning rules) is pure Java.

> Status (2026-11-05): Milestones 1-2 code complete; **Mac verification pending**. Milestone 3 tools written and tested on
> *synthetic* data only. No real dataset, model or metric exists yet. See `HANDOFF.md`, `DECISIONS.md`, `docs/MILESTONES.md`.

## Mac quick start (Apple Silicon)

```bash
# 1. place the package (keeps your current folder as a timestamped backup; nothing is deleted)
cd ~/Projects
[ -e marine-plastic ] && mv marine-plastic "marine-plastic-before-v2-$(date +%Y%m%d-%H%M)"
unzip -q ~/Downloads/marine-plastic-m2-v2.zip
cd ~/Projects/marine-plastic
shasum -a 256 -c MANIFEST.sha256 | grep -v ': OK$' ; echo "(no lines above = package intact)"

# 2. minimal setup (installs only inside this folder; safe to re-run)
bash setup.sh --minimal

# 3. tests, including the real Java ONNX smoke test
source env.sh
bash scripts/run_tests.sh

# 4. only if something failed: one diagnostics file to send to ChatGPT/Claude
bash scripts/collect_diagnostics.sh
```
After `--minimal` passes, run `bash setup.sh` once (adds PyTorch etc.; it is safe to re-run and skips what is done).

`setup.sh` options: `--minimal` (no PyTorch), `--check` (report only; creates just `logs/`), `--skip-smoke`, `--skip-java-deps`.

### What setup changes (all inside this folder) and what it never touches
Changes: `.tools/` (JDK 21.0.13+9, uv 0.13.7, private Python 3.12), `.venv/`, `java/lib/` (ONNX Runtime jar), `logs/`, `build/`,
empty folders with `.gitkeep`, and a git repo (`git init` + one commit if your git name/email are set; otherwise it tells you the 2 commands).
Never touches: system Java 17 / Python 3.13, Homebrew, `~/.zshrc` and other dotfiles, global PATH, global git config. No sudo.
Wrong-version leftovers are renamed `*.old-<time>`, never deleted.

### Expected success output (Mac, `--minimal`) - abridged
```
[ OK ] macOS 26.2 (build ...), arm64
[ OK ] checksum verified: OpenJDK21U-jdk_aarch64_mac_hotspot_21.0.13_9.tar.gz
[ OK ] JDK binary is arm64 (native Apple Silicon)
[ OK ] checksum verified: uv-aarch64-apple-darwin.tar.gz
[ OK ] Python environment verified
[ OK ] jar downloaded; SHA-256 matches the pin
[ OK ] native libraries present: ai/onnxruntime/native/osx-aarch64/libonnxruntime.dylib
[ OK ] native libraries present: ai/onnxruntime/native/osx-aarch64/libonnxruntime4j_jni.dylib
[python] builder=onnx  onnxruntime=1.20.0
[python] PASS
[java] java.version=21.0.13 ... os.arch=aarch64
[java] max|diff| batch=2: ~1e-07   batch=1: ~1e-07 ...
[java] PASS (worst abs diff ...)
[ OK ] ONNX smoke test PASSED (Python and Java agree within tolerance)
[ OK ] SETUP COMPLETE
```
`run_tests.sh` then prints `28 passed, 1 skipped` (the skip is the PyTorch test, which runs after the full install: 29 passed) and `all tests passed`.
Differences of ~1e-7 are normal; the pass limit is `|diff| <= 1e-5 + 1e-4*|expected|`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Xcode Command Line Tools missing` | `xcode-select --install`, then re-run setup |
| `running under Rosetta` | Terminal -> Get Info -> untick "Open using Rosetta"; reopen; re-run |
| `SHA-256 mismatch` | Never bypass. Re-run once; if it repeats run `bash scripts/collect_diagnostics.sh` and report |
| `jar is missing required native libraries` | Stop. Send diagnostics; the pre-agreed fallback is the HTTP detector (DECISIONS D-07) |
| `UnsatisfiedLinkError` / `Could not initialize class ai.onnxruntime...` in the Java step | Same as above; send diagnostics |
| macOS "cannot verify developer" dialog for a downloaded binary | Send the text; do not disable Gatekeeper globally |
| `git` commit skipped | Run the two `git config --global ...` lines setup prints, then `git add -A && git commit -m "chore: initial scaffold"` |
| Want a clean slate | Rename `.tools` and `.venv` (setup recreates them) |

## Layout
```
setup.sh env.sh config/toolchain.env      environment (one command)      MANIFEST.sha256   package integrity
requirements/                             exact-pinned Python deps
ml/smoke/                                 M2 export + parity reference
ml/data/                                  M3 dataset tools (inspect, near-duplicates, splits, tiles, fetch, prepare)
java/app/ java/fundamentals/              Java: ONNX smoke test now; app + fundamentals later
scripts/                                  run_tests.sh smoke_onnx.sh fetch_java_deps.sh collect_diagnostics.sh check_env.py
notebooks/01_bepli_colab_inspection.ipynb M3 on Colab (do NOT run on the Mac: 8 GB)
tests/python/                             pytest suite   configs/data.yaml   docs/   data/ models/ experiments/ reports/
```

## Evidence levels
G generated -> S sandbox-tested (Linux) -> M tested on the Mac -> D tested on real data. Nothing is "complete" without M (or D where relevant).
