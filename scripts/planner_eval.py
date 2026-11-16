import gzip, json, math, pathlib, sys, time
import numpy as np
import onnxruntime as ort

SKIP_PARITY = "--skip-parity" in sys.argv   # test hook only; the shell wrapper never passes it
t0 = time.time()
R, C, T, NC, NF = 6, 8, 48, 13, 15
NAMES = ["pet_bottle", "other_bottle", "plastic_bag", "box_shaped_case", "other_container", "rope", "other_string",
         "fishing_net", "buoy", "other_fishing_gear", "styrene_foam", "others", "fragment"]
WEIGHT = {"fishing_net": 3.0, "other_fishing_gear": 3.0, "buoy": 2.5, "pet_bottle": 2.0, "other_bottle": 2.0, "plastic_bag": 2.0,
          "box_shaped_case": 2.0, "other_container": 2.0, "rope": 2.0, "other_string": 1.5, "styrene_foam": 1.5, "others": 1.0, "fragment": 0.5}
WV = np.array([WEIGHT[n] for n in NAMES], dtype=np.float64)
KS = (3, 5, 8)
BOOT = 2000
METHODS = ["oracle", "random", "heuristic", "learned"]
LABEL = {"oracle": "oracle (ground-truth priority)", "random": "random", "heuristic": "heuristic (class weight x confidence)", "learned": "learned (transformer)"}
JAVA_REF = {"heuristic": (3, 9.243, 12.280), "learned": (7, 13.064, 24.584)}      # dashboard self-test, 003278.png, YOLO11m, 6x8, threshold 1.0
CELLD_REF = {"learned": 0.591, "heuristic": 0.524, "random": 0.095}                # Cell D top-5 overlap on the same 610 test images

with gzip.open("models/route/route_data.json.gz", "rt") as f:
    DATA = json.load(f)
TEST = DATA["test"]
sess = ort.InferenceSession("models/route/route_model.onnx", providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name
print(f"test images: {len(TEST)} | route model input {IN} {sess.get_inputs()[0].shape}")


def tile_index(cx, cy, w, h):
    c = min(C - 1, max(0, int(math.floor(cx / w * C))))
    r = min(R - 1, max(0, int(math.floor(cy / h * R))))
    return r * C + c


def features(rec, cmin=0.10):
    f = np.zeros((T, NF), np.float32)
    for x1, y1, x2, y2, cf, k in rec["dets"]:
        if cf < cmin:
            continue
        t = tile_index((x1 + x2) / 2, (y1 + y2) / 2, rec["w"], rec["h"]); k = int(k)
        f[t, k] += cf; f[t, NC] += 1; f[t, NC + 1] = max(f[t, NC + 1], cf)
    return np.log1p(f)


def heuristic_scores(rec, conf=0.25):
    s = np.zeros(T); n = np.zeros(T, dtype=int)
    for x1, y1, x2, y2, cf, k in rec["dets"]:
        if cf < conf:
            continue
        t = tile_index((x1 + x2) / 2, (y1 + y2) / 2, rec["w"], rec["h"])
        s[t] += WV[int(k)] * cf; n[t] += 1
    return s, n


def gt_priority(rec):
    y = np.zeros(T)
    for x1, y1, x2, y2, k in rec["gt"]:
        y[tile_index((x1 + x2) / 2, (y1 + y2) / 2, rec["w"], rec["h"])] += WV[int(k)]
    return y


def learned_priority(rec):
    return sess.run(None, {IN: features(rec)[None]})[0][0].astype(np.float64)


# ---- planner: the same algorithms as java/app/.../CleanupPlanner.java (tile units, entry tile (0,0)) ----
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def length(route):
    s, prev = 0.0, (0, 0)
    for t in route:
        s += dist(prev, t); prev = t
    return s


def nearest_neighbour(sel):
    rem, out, cur = list(sel), [], (0, 0)
    while rem:
        best, bd = 0, float("inf")
        for i, t in enumerate(rem):
            dd = dist(cur, t)
            if dd < bd - 1e-12:
                bd, best = dd, i
        cur = rem.pop(best); out.append(cur)
    return out


def two_opt(route):
    L = [(0, 0)] + list(route); n = len(L); improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                before = dist(L[i - 1], L[i]) + (dist(L[j], L[j + 1]) if j + 1 < n else 0.0)
                after = dist(L[i - 1], L[j]) + (dist(L[i], L[j + 1]) if j + 1 < n else 0.0)
                if after < before - 1e-9:
                    L[i:j + 1] = L[i:j + 1][::-1]; improved = True
    return L[1:]


def serpentine(sel):
    return sorted(sel, key=lambda t: (t[0], t[1] if t[0] % 2 == 0 else -t[1]))


def dashboard_plan(score, count, thr, require_count):
    sel = [(t // C, t % C) for t in range(T) if score[t] >= thr and (not require_count or count[t] > 0)]
    return len(sel), length(two_opt(nearest_neighbour(sel))), length(serpentine(sel))


# ---- gate: the Python port must reproduce the Java dashboard numbers on 003278.png ----
rec = next((r for r in TEST if r["file"] == "003278.png"), None)
if rec is None and not SKIP_PARITY:
    sys.exit("PARITY: FAIL (003278.png not found in the test records)")
if rec is not None and not SKIP_PARITY:
    sh, ch = heuristic_scores(rec)
    sl = np.expm1(np.maximum(0.0, learned_priority(rec)))
    got = {"heuristic": dashboard_plan(sh, ch, 1.0, True), "learned": dashboard_plan(sl, None, 1.0, False)}
    ok = True
    for k in ("heuristic", "learned"):
        g, e = got[k], JAVA_REF[k]
        good = g[0] == e[0] and abs(g[1] - e[1]) < 0.002 and abs(g[2] - e[2]) < 0.002
        ok &= good
        print(f"parity {k:9s}: python selected={g[0]} planned={g[1]:.3f} serpentine={g[2]:.3f} | java selected={e[0]} planned={e[1]:.3f} serpentine={e[2]:.3f} -> {'OK' if good else 'MISMATCH'}")
    if not ok:
        sys.exit("PARITY: FAIL (the Python port does not reproduce the Java dashboard; send me this output)")
    print("PARITY: PASS (Python planner reproduces the Java dashboard on 003278.png)")

# ---- evaluation set: same rule as the transformer ranking study (at least 5 ground-truth tiles) ----
Y = {i: gt_priority(r) for i, r in enumerate(TEST)}
EV = [i for i in range(len(TEST)) if (Y[i] > 0).sum() >= 5]
print(f"evaluation images: {len(EV)} of {len(TEST)} (the ranking study used 610)")
if len(EV) != 610:
    print("WARNING: expected 610 evaluation images")
HEUR = {i: heuristic_scores(TEST[i])[0] for i in EV}
LEARN = {i: learned_priority(TEST[i]) for i in EV}

rows = {(K, m): [] for K in KS for m in METHODS}
for i in EV:
    y = Y[i]
    noise = np.random.default_rng(10000 + i).random(T) * 1e-9
    scores = {"oracle": y, "random": np.random.default_rng(20000 + i).random(T), "heuristic": HEUR[i], "learned": LEARN[i]}
    for K in KS:
        o_sum = float(np.sort(y)[::-1][:K].sum())
        o_top = set(np.argsort(-(y + noise), kind="stable")[:K].tolist())
        for m in METHODS:
            top = np.argsort(-(np.asarray(scores[m], dtype=np.float64) + noise), kind="stable")[:K]
            sel = sorted(int(t) for t in top)
            tiles = [(t // C, t % C) for t in sel]
            opt = length(two_opt(nearest_neighbour(tiles))); ser = length(serpentine(tiles))
            rows[(K, m)].append((float(y[sel].sum() / o_sum), float((y[sel] > 0).mean()), len(set(sel) & o_top) / K, opt, ser))
A = {k: np.array(v) for k, v in rows.items()}


def boot(v, seed):
    v = np.asarray(v, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(v), (BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def verdict(a, b, t):
    if t[1] > 0:
        return f"{a} captures more priority than {b} (95% interval excludes 0)"
    if t[2] < 0:
        return f"{b} captures more priority than {a} (95% interval excludes 0)"
    return f"no clear difference between {a} and {b} (95% interval includes 0)"


out = {"n_eval": len(EV), "n_test": len(TEST), "K": list(KS), "bootstrap": BOOT, "results": {}, "paired_capture": {}}
md = []
w = md.append
w("# Planner evaluation (generated by scripts/planner_eval.py; do not edit by hand)\n")
w(f"Test split, {len(EV)} of {len(TEST)} images with at least 5 ground-truth tiles (6x8 grid, YOLO11m detections at confidence 0.10 and above, transformer trained on the val split only).\n")
w("Selection rule: take the K highest-scoring tiles of each image (ties broken by a fixed seeded noise). Priority of a tile = sum of the class weights of the ground-truth boxes whose centre falls in it (same weights as the heuristic planner). Intervals are 95% bootstrap intervals over images (2000 resamples).\n")
w("## 1. How much ground-truth priority do the selected tiles capture?\n")
w("Priority captured = priority in the K selected tiles divided by the priority in the best possible K tiles (1.000 is perfect).\n")
w("| K | method | priority captured (mean [95% CI]) | selected tiles with any ground-truth priority | overlap with oracle top-K |")
w("|---:|---|---:|---:|---:|")
for K in KS:
    out["results"][str(K)] = {}
    for m in METHODS:
        a = A[(K, m)]
        cap = boot(a[:, 0], 100 * K + METHODS.index(m))
        r_ = {"capture": cap, "precision": float(a[:, 1].mean()), "overlap": float(a[:, 2].mean()),
              "opt_len": float(a[:, 3].mean()), "ser_len": float(a[:, 4].mean())}
        d_ = a[:, 4] - a[:, 3]
        r_["ser_minus_opt"] = boot(d_, 300 * K + METHODS.index(m))
        r_["opt_shorter_pct"] = float((d_ > 1e-9).mean() * 100); r_["equal_pct"] = float((abs(d_) <= 1e-9).mean() * 100); r_["opt_longer_pct"] = float((d_ < -1e-9).mean() * 100)
        out["results"][str(K)][m] = r_
        w(f"| {K} | {LABEL[m]} | {fmt(cap)} | {r_['precision']:.3f} | {r_['overlap']:.3f} |")
w("\n### Paired differences in priority captured (same images)\n")
w("| K | comparison | difference (mean [95% CI]) | reading |")
w("|---:|---|---:|---|")
for K in KS:
    out["paired_capture"][str(K)] = {}
    for a_, b_ in (("learned", "heuristic"), ("learned", "random"), ("heuristic", "random")):
        t = boot(A[(K, a_)][:, 0] - A[(K, b_)][:, 0], 500 * K + len(a_) + len(b_))
        out["paired_capture"][str(K)][f"{a_}_minus_{b_}"] = t
        w(f"| {K} | {a_} minus {b_} | {fmt(t)} | {verdict(a_, b_, t)} |")
w("\n## 2. Route over the selected tiles: optimised route vs serpentine sweep\n")
w("Both routes visit exactly the same K tiles, start at the entry tile (0,0) and are measured in tile units (not metres or minutes). Optimised = nearest neighbour followed by 2-opt (the dashboard planner). Serpentine = row-by-row sweep restricted to those tiles.\n")
w("| K | method | optimised length | serpentine length | serpentine minus optimised (mean [95% CI]) | optimised shorter / equal / longer (% of images) |")
w("|---:|---|---:|---:|---:|---:|")
for K in KS:
    for m in METHODS:
        r_ = out["results"][str(K)][m]
        w(f"| {K} | {LABEL[m]} | {r_['opt_len']:.2f} | {r_['ser_len']:.2f} | {fmt(r_['ser_minus_opt'], 2)} | {r_['opt_shorter_pct']:.1f} / {r_['equal_pct']:.1f} / {r_['opt_longer_pct']:.1f} |")
allp = all(out["results"][str(K)][m]["ser_minus_opt"][1] > 0 for K in KS for m in METHODS)
w("\n" + ("In every row the 95% interval of (serpentine minus optimised) is above 0, so the optimised route is shorter on average." if allp
          else "In at least one row the 95% interval of (serpentine minus optimised) includes 0 or is negative; read the table row by row."))
w("The optimised route is not guaranteed to beat the serpentine on every image (2-opt starts from a nearest-neighbour tour); the last column shows how often it does.\n")
w("## 3. Consistency checks\n")
w("| check | this script | earlier result | status |")
w("|---|---:|---:|---|")
cons = {}
for m in ("learned", "heuristic", "random"):
    v_, ref = out["results"]["5"][m]["overlap"], CELLD_REF[m]
    st = "OK" if abs(v_ - ref) <= 0.03 else "WARN (different tie-breaking or exported seed)"
    cons[m] = {"value": v_, "reference": ref, "status": st}
    w(f"| top-5 overlap, {m} | {v_:.3f} | {ref:.3f} (Cell D) | {st} |")
out["consistency"] = cons
if not SKIP_PARITY:
    w("| Python planner vs Java dashboard on 003278.png | reproduced | Java self-test numbers | PASS |")
w("\n## 4. Limitations\n")
w("- Priority comes from ground-truth boxes weighted by assumed class weights, so this measures how well the planner finds the tiles that matter under that definition, not real cleanup effort.")
w("- Route lengths are in tile units from a fixed entry tile; they say nothing about walking time, terrain or GPS.")
w("- The transformer was trained on val-split detections and evaluated on test; the YOLO11m detector was trained on the train split. One training run of each.")
w("- The comparison covers one fixed rule (top-K tiles). The dashboard threshold rule selects a different number of tiles per image and is not evaluated here.")
text = "\n".join(md) + "\n"
pathlib.Path("results").mkdir(exist_ok=True); pathlib.Path("docs").mkdir(exist_ok=True)
json.dump(out, open("results/planner_eval.json", "w"), indent=1)
pathlib.Path("docs/PLANNER_EVAL.md").write_text(text)
print("\n" + text)
print(f"wrote results/planner_eval.json and docs/PLANNER_EVAL.md | {time.time() - t0:.1f} s")
print("PLANNER EVAL: DONE")
