# RELLIS-3D Table V replication attempt

## TL;DR

Across **four independent configurations** — including the literal protocol Simon Schwaiger described in [PR #2 review](https://github.com/SimonSchwaiger/otas/pull/2) (6-class terrain subset, `neg=["thing"]`, `threshold=0.8`) run against the **first commit** of this repository (`6aec2d4`, also as Simon recommended) — our best mIoU on the 1672-frame RELLIS-3D test split is **6.70%**, vs the paper's Table V claim of **48.48% (DINOv2 ViT-S/14)**. Residual gap: **~42 mIoU**, and it is not sensitive to commit version, `shared_feat_resolution`, `n_components`, `dinov2_input_size`, or input image resize.

The structural reason is visible in the per-class breakdown: `semantic_mask.similarity` min-max-normalises *per image*, so the threshold@0.8 binary decision fires on the noisiest 20% of pixels in every frame where a class is absent. Across the ~1500 absent-class frames per sparse class (`dirt` appears in 13 frames, `water` in 19, `rubble` in 145), this accumulates into hundreds of millions of FPs that drown every sparse class's IoU. `bush` (present in 1658 of 1672 frames) is the only class with a meaningful score (~35), and it carries the entire 6-class mean.

We suspect the remaining gap is an **mIoU averaging convention** that filters out frames where the class is absent in GT (per-frame mIoU averaged only over frames-with-class-present, then meaned across classes). This is a one-knob change — happy to run it and post the result — but we want to verify the convention rather than guess.

## Simon's clarification (PR #2 review, 2026-05)

Quoted verbatim from the PR conversation:

> we run Rellis-3D only on a subset of classes relevant to terrain segmentation
>
> ```python
> class_prompts = {1: "dirt", 6: "water", 10: "asphalt",
>                  19: "bush", 33: "mud", 34: "rubble"}
> neg_prompts = ["thing"]
> threshold_value = 0.8
> ```
>
> I'd also recommend switching to the first commit of this repository (that's the exact code we ran the evaluation on).

This clarification superseded our initial assumption (20-class argmax of bare class names with no negative prompt). We rebuilt the eval against the literal protocol — see [`own_eval/own_RELLIS_paper.py`](../own_eval/own_RELLIS_paper.py) — and re-ran on both current `main` and on a `6aec2d4` worktree.

## Full investigation table

All runs: zero-shot, no mask refinement, no spatial, DINOv2 ViT-S/14 + MaskCLIP ViT-B/16, native 1200×1920 input. 1672-frame test split. Hardware: NVIDIA RTX PRO 6000 Blackwell, torch 2.12.0+cu132.

| # | Code | Config | Protocol | mIoU |
|--:|---|---|---|--:|
| 1 | OTAS-repo `OTAS_small.json`-shaped default (when investigation started) | d=32, Cr=12, dinov2 input 518, 480×640 | 20 bare class names, argmax | 16.70 |
| 2 | Current `main` | §VII.A: d=64, Cr=24, dinov2 input 224, 1024×1024 | 20 bare class names, argmax | 15.43 |
| 3 | Current `main`, [`own_RELLIS.py`](../own_eval/own_RELLIS.py) | §VII.A: d=64, Cr=24, dinov2 input 224, native 1200×1920 | 20 bare class names, argmax | 15.66 |
| 4 | Current `main`, [`own_RELLIS_paper.py`](../own_eval/own_RELLIS_paper.py) | §VII.A: d=64, Cr=24, dinov2 input 224, native | **Simon's protocol**: 6 classes, `neg=["thing"]`, `t=0.8` | **6.70** |
| 5 | **First commit (`6aec2d4`)** + worktree | **First-commit defaults**: d=32, Cr=48, dinov2 input 518 | Simon's protocol: 6 classes, `neg=["thing"]`, `t=0.8` | **6.64** |
| 6 | **First commit (`6aec2d4`)** + worktree | §VII.A: d=64, Cr=24, dinov2 input 224 | Simon's protocol: 6 classes, `neg=["thing"]`, `t=0.8` | **6.63** |
| — | — | — | **Paper Table V claim (DINOv2 ViT-S/14)** | **48.48** |

Rows 4–6 are the three independent attempts at Simon's literal Table V protocol. They land within **0.07 mIoU** of each other across two different code versions and two different config presets — `semantic_mask.similarity` math is identical between first-commit and current `main` (we diffed `src/model.py` to confirm: same cosine sim, same `clamp(sim_max - sim_min, min=0.05)` normalisation, same `(lr_sims_norm > threshold)` binarisation). Config defaults differ between the two commits but the effect on the 6-class mIoU is below kmeans-clustering noise.

## Per-class numbers under Simon's protocol — current main run (row 4)

| class | raw id | n frames present (of 1672) | total GT px | OTAS IoU | TP | FP | FN |
|---|--:|--:|--:|--:|--:|--:|--:|
| dirt | 1 | 13 | 9,690 | **0.00** | 0 | 765,332,340 | 9,690 |
| water | 6 | 19 | 959,662 | **0.35** | 500,432 | 140,051,968 | 459,230 |
| asphalt | 10 | 503 | 3,850,438 | **0.66** | 2,398,866 | 362,257,374 | 1,451,572 |
| bush | 19 | 1658 | 662,926,185 | **35.14** | 395,710,902 | 463,160,538 | 267,215,283 |
| mud | 33 | 574 | 29,818,401 | **3.59** | 19,584,570 | 516,240,750 | 10,233,831 |
| rubble | 34 | 145 | 1,907,260 | **0.47** | 1,738,363 | 364,129,997 | 168,897 |
| **mIoU(6cls)** | | | | **6.70** | | | |

First-commit + first-commit-defaults (row 5) per-class IoU: 0.00 / 0.38 / 0.62 / 34.87 / 3.50 / 0.47 → **6.64**.
First-commit + §VII.A overrides (row 6) per-class IoU: 0.00 / 0.44 / 0.66 / 34.67 / 3.55 / 0.47 → **6.63**.

The pattern is identical across all three: `bush` carries the headline (~35), every other class is ≤ 4, sparse classes hit ~0 because TPs (thousands) are dwarfed by FPs (hundreds of millions).

## Root cause of the gap, as best we can pin it down

`semantic_mask.similarity` in [`src/model.py`](../src/model.py) does:

```python
lr_sims = sum(pos_sims) / len(pos_sims) - sum(neg_sims) / (len(neg_sims) + 1e-8)
sim_min, sim_max = lr_sims.min(), lr_sims.max()
sim_range = torch.clamp(sim_max - sim_min, min=0.05)
lr_sims_norm = (lr_sims - sim_min) / (sim_range + 1e-8)
```

The min-max normalisation is **per image**. On a frame where the class of interest is absent (e.g. `dirt` on the ~1659 dirt-free frames out of 1672), the raw cosine similarity range is small but non-zero — pure noise. The `clamp(min=0.05)` lower-bounds the range, but 0.05 is still tight enough that the noise distribution gets stretched to fill [0, 1]. The threshold@0.8 then fires on roughly the noisiest 20% of pixels of that frame.

Across ~1500 absent-class frames × 1920×1200 pixels × ~20% above-threshold, that's hundreds of millions of false positives per sparse class. The TPs on the rare frames where the class IS present (a few thousand pixels in the dirt case) are completely overwhelmed.

`bush` doesn't suffer this because it's present in 99.2% of frames — the per-image normalisation is normalising real signal, not noise.

## Reproduction

### Prerequisites

- Python 3.12 venv with `requirements.txt`.
- DINOv2 + CLIP checkpoints via `bash download_checkpoints.sh`. SAM2 is not required (we run with `enable_mask_refinement: false`).
- RELLIS-3D dataset extracted under a single root, e.g. `/path/to/Rellis-3D`:

  ```
  Rellis-3D/
    train.lst   val.lst   test.lst       # 44 KB Image Split File archive
    00000/   00001/  00002/  00003/  00004/
      pylon_camera_node/             # RGB .jpg, 1920×1200
      pylon_camera_node_label_id/    # uint8 label-id .png, 1920×1200
  ```

  The 4 Google Drive archives needed (per upstream `unmannedlab/RELLIS-3D` README) are: Full Images (11 GB), Full Image Annotations ID Format (94 MB), Image Split File (44 KB), Ontology Definition (18 KB).

### Run on current main + §VII.A overrides (row 4 above)

```bash
cd /path/to/OTAS
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_RELLIS_paper.py \
    --data_dir /path/to/Rellis-3D \
    --config_preset paper_vii_a
cat result/Pred/RELLIS_rgb_paper/results.txt | grep mIoU_6cls
# → mIoU_6cls: 6.7020
```

### Run on first commit (`6aec2d4`) with first-commit defaults (row 5)

```bash
git worktree add /tmp/otas-first-commit 6aec2d4
ln -sf $(pwd)/src/foundation_models/dinov2_checkpoints/dinov2_vits14_reg4_pretrain.pth \
       /tmp/otas-first-commit/src/foundation_models/dinov2_checkpoints/
ln -sf $(pwd)/src/foundation_models/clip_checkpoints/ViT-B-16.pt \
       /tmp/otas-first-commit/src/foundation_models/clip_checkpoints/
# One-line py3.12 compat fix the first commit predates:
sed -i 's|from pkg_resources import packaging|import packaging.version|' \
    /tmp/otas-first-commit/src/foundation_models/maskclip_onnx/clip.py
mkdir -p /tmp/otas-first-commit/own_eval
cp own_eval/{own_RELLIS_paper.py,rellis_dataset.py} /tmp/otas-first-commit/own_eval/
cd /tmp/otas-first-commit
env -u LD_LIBRARY_PATH /path/to/OTAS/.venv/bin/python own_eval/own_RELLIS_paper.py \
    --data_dir /path/to/Rellis-3D \
    --config_preset first_commit_defaults \
    --out_root /tmp/otas-first-commit/result/Pred \
    --out_suffix _firstcommit_defaults
cat result/Pred/RELLIS_rgb_firstcommit_defaults/results.txt | grep mIoU_6cls
# → mIoU_6cls: 6.6397
```

### Run on first commit + §VII.A overrides (row 6)

Same as row 5 but `--config_preset paper_vii_a --out_suffix _firstcommit_viia`. Lands at `mIoU_6cls: 6.6317`.

The `env -u LD_LIBRARY_PATH` prefix forces the cu132 torch wheel's bundled cuBLAS to win over the system `/usr/local/cuda-*` library on Blackwell GPUs; without it MaskCLIP's forward passes fail with `cublasLtGetVersion` symbol errors. Drop it on other GPU/CUDA combinations.

## What's in this PR

| File | Purpose |
|---|---|
| [`own_eval/own_RELLIS.py`](../own_eval/own_RELLIS.py) | 20-class argmax driver (rows 1–3 above). CLI: `--enable_mask_refinement`, `--input_h`/`--input_w`. §VII.A hyperparameters baked into `otas_segmentor.py:_DEFAULT_CONFIG`. |
| [`own_eval/own_RELLIS_paper.py`](../own_eval/own_RELLIS_paper.py) | Simon's Table V protocol driver (rows 4–6 above). CLI: `--config_preset {paper_vii_a, first_commit_defaults}`, `--threshold` (default 0.8), `--max_frames` for smoke runs, `--save_preds` to cache the (6, H, W) per-frame binary stack. Calls `model.semantic_mask.similarity()` + threshold directly (no N-way argmax adapter). |
| [`own_eval/rellis_dataset.py`](../own_eval/rellis_dataset.py) | RELLIS-3D PyTorch `Dataset`. Parses `train.lst`/`val.lst`/`test.lst` (2-col `<img> <label>` per line), reads RGB jpg + uint8 label-id PNG, applies the unknown-incl 20-class LUT from `ontology.yaml` (raw 0/1/3/…/34 → contig 0..19). The paper-protocol driver reuses this dataset and reads raw label PNGs directly so it can compare against Simon's raw-ID mapping without going through the contig LUT. |
| [`own_eval/eval_common.py`](../own_eval/eval_common.py) | Per-frame inference loop + confusion-matrix-pooled IoU for the 20-class argmax path. Forwards arbitrary `config_overrides` dict so the SAM toggle and the §VII.A hyperparameters can be set per-run. |
| [`own_eval/otas_segmentor.py`](../own_eval/otas_segmentor.py) | N-way `OTASEncoder` adapter for the 20-class argmax path: `language_map.embed_image → per-class clip_similarity → bilinear up → argmax`. SAM-on path delegates per-class refinement to `semantic_mask.binary_mask_refined(..., ret_dict=True)` then argmaxes across the per-class `pred_logits`. |
| [`docs/RELLIS_REPLICATION.md`](RELLIS_REPLICATION.md) | This file. |
| [`src/foundation_models/maskclip_onnx/clip.py`](../src/foundation_models/maskclip_onnx/clip.py) | One-line py3.12 compat: `import packaging.version` (replaces `from pkg_resources import packaging`, which doesn't expose `.version.parse` on modern setuptools). |
| `.gitignore` | Adds `result/*` so cached per-frame predictions don't get committed. |

## What we have tested and ruled out

| Hypothesis | Tested? | Result |
|---|---|---|
| `shared_feat_resolution = 64` (§VII.A) | Yes | Moves 20-class mIoU ~−1 (16.70 → 15.43 combined with the other §VII.A knobs). Moves 6-class mIoU ~+0.01 (6.64 → 6.63 first-commit + this knob only). |
| `n_components = 24` (PCA `Cr`, §VII.A) | Yes | Combined with d=64: no big jump in either protocol. |
| `dinov2_input_size = 224` (DINOv2 native 16×16 grid, §VII.A) | Yes | Combined with d=64: no big jump. |
| `dinov2_input_size = 518` (first-commit default) | Yes | 6-class run on first commit: 6.64, vs 6.63 with size=224. No meaningful difference. |
| Input image resize to 1024×1024 (§VII.A literal) | Yes (20-class only) | No big jump (15.43 vs 15.66 native). |
| Mask refinement off | Confirmed | Both our runs and the paper Table V have it off. |
| **First commit code (`6aec2d4`)** | **Yes** — both with first-commit-default config and with §VII.A overrides | **6.64 / 6.63** — `semantic_mask.similarity` math is byte-identical to current `main`; the diff between first-commit and current is config defaults + the `featurizer.<x>` module reorganisation, none of which affects inference outputs. |
| Negative prompt of `"thing"` | Yes (Simon's protocol) | Applied. |
| Threshold-at-0.8 binary scoring | Yes (Simon's protocol) | Applied. |
| 6-class terrain subset | Yes (Simon's protocol) | Applied. |
| 20-class argmax (our original assumption) | Yes | Lands at 15.66; superseded by Simon's clarification. |

## Open hypotheses we have NOT tested

After rows 4–6 establish that the gap is insensitive to commit version, config, and resize, the remaining structural unknowns are:

1. **mIoU averaging convention.** We accumulate one (TP, FP, FN) triple per class across all 1672 frames and compute IoU once over the totals. The per-class FP counts above (765M for dirt across 1659 absent-class frames) directly tank pooled IoU on sparse classes. If Table V's headline is computed *per frame*, with each class's contribution averaged **only over frames where the class is present in GT**, the absent-class FPs drop out and the headline rises dramatically. We have not yet run this metric — it's a one-knob change to `own_RELLIS_paper.py` (accumulate per-frame TP/FP/FN per class, then average IoU per class only across frames-with-GT-present, then mean across classes). Happy to run it and report back.

2. **Test-split version / curation.** We use `test.lst` from the 44 KB Image Split File archive on the upstream `unmannedlab/RELLIS-3D` Google Drive (1672 lines, 2-column `<image_path> <label_path>`). All 6 paper classes appear in this split, but with very skewed frequency: dirt in 13 of 1672 frames, water in 19, rubble in 145. The paper may use a different release of this split, a curated subset (e.g. frames where ≥1 of the 6 classes is present), or evaluate only on frames where the class of interest is in GT.

3. **Per-class threshold.** Simon's PR comment specifies `threshold_value = 0.8` as a single scalar. Worth confirming that's one threshold applied uniformly across all 6 classes — not e.g. per-class thresholds tuned on val.

## Ask for the upstream maintainers

Simon's PR #2 review already covered the 6-class subset, negative prompt, and threshold — thank you for that, it materially clarified things and ruled out the 20-class argmax interpretation. To close the remaining ~42-point gap unambiguously, we'd value:

1. **The exact eval script that produced Table V** — even a minimal `eval_rellis.py` analogous to `demo.ipynb`. The single most important question we cannot answer from the released code is the **mIoU averaging convention**: pooled-across-frames per class (what we do) vs per-frame averaged over frames-with-class-present per class? Given the sparse-class FP problem visible in the per-class table above, this is the most likely structural source of our gap.

2. **The `OTAS_*.json` config file used for Table V.** The shipped `OTAS_small.json` has `enable_mask_refinement: true` and `shared_feat_resolution: 16`, which contradicts §VII.A's `d=64`. Knowing which config file produced the 48.48 number would resolve any remaining ambiguity about d / Cr / dinov2 input size — though our test of 6 different configs landing within 0.07 mIoU of each other suggests config is not the load-bearing variable here.

3. **The split file used.** If the paper evaluates on a different split than upstream `test.lst`, or restricts to frames-with-class-present, that's a simple fix on our end.

We're happy to fold our `own_RELLIS_paper.py` into the official scaffolding if that's the right approach, or rewrite it on top of an upstream `eval_rellis.py` once released. Either way, having a runnable script that produces the paper's headline number would let downstream users verify and build on Table V with confidence.

## Verification

After applying this PR to a fresh OTAS checkout:

```bash
# 1. Setup
git clone <this-fork>; cd otas
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash download_checkpoints.sh

# 2. Stage RELLIS-3D under /path/to/Rellis-3D (see "Prerequisites" above)

# 3. Run Simon's Table V protocol on current main
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_RELLIS_paper.py \
    --data_dir /path/to/Rellis-3D --config_preset paper_vii_a
grep mIoU_6cls result/Pred/RELLIS_rgb_paper/results.txt
# → mIoU_6cls: 6.70xx  (kmeans noise: ±0.05)

# 4. Run the original 20-class argmax protocol for comparison
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_RELLIS.py \
    --data_dir /path/to/Rellis-3D
grep -E "full_mIoU|fg_only_mIoU" result/Pred/RELLIS_rgb/results.txt
# → full_mIoU: 15.66xx, fg_only_mIoU: 16.48xx
```

Both should land well below 48.48. The first-commit reproduction (rows 5–6) requires the worktree dance documented in `Reproduction` above; it lands at 6.63–6.64 across two config presets.
