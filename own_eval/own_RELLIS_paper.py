# OTAS Table V replication on RELLIS-3D under the protocol Simon Schwaiger confirmed
# in https://github.com/SimonSchwaiger/otas/pull/2 (review comment, 2026-05).
#
# Protocol — verbatim from Simon's PR comment:
#     class_prompts = {1: "dirt", 6: "water", 10: "asphalt",
#                      19: "bush", 33: "mud", 34: "rubble"}   # raw RELLIS IDs
#     neg_prompts = ["thing"]
#     threshold_value = 0.8
#
# This is a 6-class binary-per-class evaluation, NOT the 20-class argmax our existing
# own_RELLIS.py runs. Each of the 6 terrain classes is scored independently as a binary
# segmentation task. For class c at every pixel:
#   1. lr_sims_norm = min-max-normalised( sim(img, "c") - sim(img, "thing") )
#   2. pred_c = (lr_sims_norm > 0.8)            # threshold-binarised, no argmax
#   3. gt_c   = (raw_label == raw_id_of_c)
#   4. TP/FP/FN accumulated independently per class across all 1672 test-split frames
#   5. mIoU = mean over 6 classes of TP / (TP + FP + FN)
#
# This differs structurally from own_RELLIS.py:
#   - own_RELLIS.py: 20 class names as positive prompts, no negative prompt, per-pixel
#     argmax across the 20 score maps, mIoU computed via 20×20 confusion matrix. Got
#     15.66 full / 16.48 fg-only mIoU on the 1672-frame test split.
#   - own_RELLIS_paper.py (this file): 6 class names looped one-by-one as pos prompt,
#     "thing" as neg prompt, threshold@0.8 per class, mIoU computed from per-class
#     binary TP/FP/FN. Target: paper's Table V claim of 48.48 mIoU.
#
# This driver calls OTAS's `semantic_mask.similarity(...)` + threshold directly rather
# than reusing the multi-class OTASEncoder adapter in otas_segmentor.py, because:
#   - The N-way argmax adapter was designed for the apples-to-apples cross-system
#     scoreboard (RADSeg / OpenRSS / OTAS all do N-way argmax). The paper's per-class
#     threshold protocol doesn't fit that mould.
#   - OTAS's `semantic_mask.similarity` already returns the min-max-normalised score
#     map in [0,1] that the threshold acts on. Reusing it verbatim guarantees we're
#     scoring against the exact normalisation OTAS itself uses internally.

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Make own_eval/ importable for rellis_dataset.
_OWN_EVAL_DIR = str(Path(__file__).resolve().parent)
if _OWN_EVAL_DIR not in sys.path:
    sys.path.insert(0, _OWN_EVAL_DIR)

from rellis_dataset import RELLIS_dataset  # reuses the existing PNG/LUT pipeline


# Simon's mapping: raw RELLIS ontology IDs -> bare class-name prompts. Order is fixed so
# the per-class IoU table is deterministic; per-class IoU mean is order-invariant.
PAPER_CLASS_PROMPTS = [
    (1,  "dirt"),
    (6,  "water"),
    (10, "asphalt"),
    (19, "bush"),
    (33, "mud"),
    (34, "rubble"),
]
NEG_PROMPTS = ["thing"]
THRESHOLD_VALUE = 0.8


def _resolve_otas():
    # Add OTAS src/ to sys.path and import model/vision_utils. Mirrors the lazy import
    # dance otas_segmentor.OTASEncoder does so the heavy DINOv2/MaskCLIP loads only fire
    # once and only after OTAS_CONFIG_PATH is set.
    src = str(Path(__file__).resolve().parent.parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import model  # noqa: E402
    import vision_utils  # noqa: E402
    return model, vision_utils


def _write_config(preset: str = "paper_vii_a", custom_overrides=None):
    # Same config-tempfile trick otas_segmentor uses: write a JSON file holding hyperparameters
    # + SAM/spatial toggles, point OTAS_CONFIG_PATH at it.
    #
    # Two presets:
    #   "paper_vii_a"           - §VII.A overrides: d=64, Cr=24, k=24, dinov2_input_size=224.
    #                             Mask refinement off, spatial off. Same overrides our
    #                             20-class own_RELLIS.py uses.
    #   "first_commit_defaults" - Only override the SAM/spatial/compilation toggles. Lets the
    #                             first-commit `src/config.py` defaults drive shared_feat_resolution
    #                             (32), n_components (48), n_clusters (24) and dinov2_input_size
    #                             (effectively 518 via dinov2_params img_size). Simon recommended
    #                             "switching to the first commit of this repository" in PR #2; this
    #                             preset is the literal default config from that commit minus the
    #                             SAM toggle (which would also load Open3D + sam2 weights).
    import json
    if preset == "paper_vii_a":
        cfg = {
            "enable_mask_refinement": False,   # Table V is SAM-off (confirmed in paper §VII.A)
            "enable_spatial": False,
            "enable_model_compilation": False,
            "dinov2_input_size": 224,          # §VII.A
            "dino_scale_factor": 2,
            "shared_feat_resolution": 64,      # §VII.A (d=64)
            "n_clusters": 24,                  # §VII.A (k=24)
            "n_components": 24,                # §VII.A (Cr=24)
            "enable_amp_autocast": True,
        }
    elif preset == "first_commit_defaults":
        cfg = {
            "enable_mask_refinement": False,   # off so we don't need SAM2 weights
            "enable_spatial": False,           # off so Open3D doesn't load
            "enable_model_compilation": False, # off; current main turns this on but it adds
                                               # 30 s of cold-start with no inference effect
            "enable_amp_autocast": True,
            # NO override of d / n_components / n_clusters / dinov2_input_size: lets the
            # first-commit src/config.py defaults win (d=32, Cr=48, k=24, dinov2_input_size
            # not in config dict so the dinov2_params img_size=518 effectively applies).
        }
    else:
        raise ValueError(f"Unknown preset: {preset!r}")
    if custom_overrides:
        cfg.update(custom_overrides)
    cfg_path = Path("/tmp") / f"otas_paper_cfg_{os.getpid()}.json"
    cfg_path.write_text(json.dumps(cfg))
    return str(cfg_path)


@torch.no_grad()
def _per_class_pred(language_map, semantic_mask_inst, pil_img, class_name, neg_prompts,
                    threshold, target_h, target_w):
    # Returns a uint8 (target_h, target_w) binary mask of where the per-class threshold
    # fires under OTAS's own normalisation. Mirrors what `binary_mask_interpolated` does
    # internally but lets us share the language_map.embed_image call across all 6
    # classes for one image, instead of paying that cost 6× per frame.
    pooled = language_map.embed_image(pil_img)              # (H_lr, W_lr, D)
    lr_sims_norm = semantic_mask_inst.similarity(
        shared_feature_map=pooled,
        pos_prompts=[class_name],
        neg_prompts=neg_prompts,
    )                                                       # (H_lr, W_lr) in [0,1]
    binary = (lr_sims_norm > threshold).to(pooled.dtype)
    binary_up = F.interpolate(
        binary.unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode="nearest",                                     # matches binary_mask_interpolated
    ).squeeze().to(torch.uint8).cpu().numpy()
    return binary_up


@torch.no_grad()
def _per_frame_six_classes(language_map, semantic_mask_inst, pil_img,
                           class_names, neg_prompts, threshold, target_h, target_w):
    # Computes all 6 per-class binary masks for one frame in a single
    # `language_map.embed_image` call (the expensive part) + 6 cheap
    # `semantic_mask.similarity` einsums + 6 NEAREST upsamples.
    # Returns (6, target_h, target_w) uint8 stack in the order of `class_names`.
    pooled = language_map.embed_image(pil_img)              # (H_lr, W_lr, D) — ONCE per frame
    out = np.zeros((len(class_names), target_h, target_w), dtype=np.uint8)
    for i, name in enumerate(class_names):
        lr_sims_norm = semantic_mask_inst.similarity(
            shared_feature_map=pooled,
            pos_prompts=[name],
            neg_prompts=neg_prompts,
        )                                                   # (H_lr, W_lr) in [0,1]
        binary = (lr_sims_norm > threshold).to(pooled.dtype)
        binary_up = F.interpolate(
            binary.unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze().to(torch.uint8).cpu().numpy()
        out[i] = binary_up
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ubuntu/mnt/rellis_3d/Rellis-3D")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_root", default="/home/ubuntu/code/OTAS/result/Pred")
    parser.add_argument("--input_h", type=int, default=None,
                        help="Optional resize height. Default: native 1200.")
    parser.add_argument("--input_w", type=int, default=None,
                        help="Optional resize width. Default: native 1920.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_VALUE,
                        help=f"Per-class binary threshold. Default {THRESHOLD_VALUE} "
                             "(Simon's PR comment).")
    parser.add_argument("--out_suffix", default="_paper",
                        help="Output dir suffix. Default '_paper' so this lands at "
                             "result/Pred/RELLIS_rgb_paper/ alongside the 20-class "
                             "argmax run at result/Pred/RELLIS_rgb/.")
    parser.add_argument("--save_preds", action="store_true",
                        help="Cache the (6, H, W) binary stack per frame. Off by "
                             "default — disk cost is ~6× the 20-class run.")
    parser.add_argument("--num_overlay_samples", type=int, default=12)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap frames evaluated. Default None (full split). "
                             "Set e.g. 5 for a smoke test.")
    parser.add_argument("--config_preset", default="paper_vii_a",
                        choices=["paper_vii_a", "first_commit_defaults"],
                        help="paper_vii_a: §VII.A overrides (d=64, Cr=24, k=24, "
                             "dinov2_input_size=224). first_commit_defaults: only "
                             "override SAM/spatial/compilation toggles; leaves "
                             "src/config.py defaults (d=32, Cr=48, dinov2 input 518) "
                             "intact. Use the latter against a 6aec2d4 worktree to "
                             "follow Simon's 'switch to the first commit' suggestion.")
    args = parser.parse_args()

    # Eager-import OTAS first so config knobs land in os.environ before model.py runs.
    cfg_path = _write_config(preset=args.config_preset)
    os.environ["OTAS_CONFIG_PATH"] = cfg_path
    model, vision_utils = _resolve_otas()
    config = model.config

    print(f"[own_RELLIS_paper] OTAS config: shared_feat_resolution="
          f"{config.get('shared_feat_resolution')}, n_clusters={config.get('n_clusters')}, "
          f"n_components={config.get('n_components')}, dinov2_input_size="
          f"{config.get('dinov2_input_size')}, enable_mask_refinement="
          f"{config.get('enable_mask_refinement')}")

    language_map = model.language_map(config=config)
    semantic_mask_inst = model.semantic_mask(config=config)

    # Dataset — reuse the 20-class RELLIS_dataset, but we only consume raw_label values
    # via the existing LUT. We bypass the contig 0..19 mapping by recomputing GT from
    # the raw label PNG inside the loop (the LUT collapses gaps to 0, so we can read
    # raw IDs directly from the PNG).
    dataset = RELLIS_dataset(
        data_dir=args.data_dir,
        split=args.split,
        input_h=args.input_h,
        input_w=args.input_w,
    )
    print(f"[own_RELLIS_paper] dataset: {len(dataset)} frames, native resize: "
          f"{args.input_h}×{args.input_w} (None means dataset-native)")

    # Per-class accumulators. Each is a length-6 array indexed by paper-class-order.
    raw_ids = np.array([rid for (rid, _) in PAPER_CLASS_PROMPTS], dtype=np.int64)
    class_names = [n for (_, n) in PAPER_CLASS_PROMPTS]
    n_cls = len(class_names)
    tp = np.zeros(n_cls, dtype=np.int64)
    fp = np.zeros(n_cls, dtype=np.int64)
    fn = np.zeros(n_cls, dtype=np.int64)

    out_dir = Path(args.out_root) / f"RELLIS_rgb{args.out_suffix}"
    overlays_dir = out_dir / "overlays"
    preds_dir = out_dir / "preds"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preds:
        preds_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(dataset)
    if args.max_frames is not None:
        n_frames = min(n_frames, args.max_frames)
        print(f"[own_RELLIS_paper] capping eval to first {n_frames} frames (--max_frames)")
    overlay_indices = set(np.linspace(0, max(n_frames - 1, 1),
                                      num=min(args.num_overlay_samples, n_frames),
                                      dtype=int).tolist())

    t0 = time.time()
    pbar = tqdm(range(n_frames), desc=f"OTAS-paper[{args.split}]")
    for idx in pbar:
        image_f, _label_contig, name = dataset[idx]
        # We need the RAW RELLIS label (not the contig 0..19 remapped one) so we can
        # compare against Simon's raw-ID mapping directly. Pull it via the dataset's
        # path bookkeeping — cheaper than re-mapping from contig back to raw.
        img_rel, label_rel = dataset.pairs[idx]
        raw_label_path = os.path.join(dataset.data_dir, label_rel)
        raw_label = np.asarray(Image.open(raw_label_path))   # uint8 (1200, 1920)
        # Honour --input_h/--input_w resize. If the dataset is in native mode (None),
        # raw_label is already the right shape.
        if args.input_h is not None and args.input_w is not None:
            raw_label = np.asarray(
                Image.fromarray(raw_label).resize((args.input_w, args.input_h),
                                                  resample=Image.NEAREST))
        target_h, target_w = raw_label.shape

        rgb = (image_f[:3].numpy().clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(rgb, mode="RGB")

        binary_stack = _per_frame_six_classes(
            language_map=language_map,
            semantic_mask_inst=semantic_mask_inst,
            pil_img=pil,
            class_names=class_names,
            neg_prompts=NEG_PROMPTS,
            threshold=args.threshold,
            target_h=target_h,
            target_w=target_w,
        )                                                    # (6, H, W) uint8

        # Per-class binary IoU accumulator.
        for c, raw_id in enumerate(raw_ids):
            gt_c = (raw_label == raw_id)
            pred_c = binary_stack[c].astype(bool)
            tp[c] += np.logical_and(gt_c, pred_c).sum()
            fp[c] += np.logical_and(~gt_c, pred_c).sum()
            fn[c] += np.logical_and(gt_c, ~pred_c).sum()

        if args.save_preds:
            # Save as a single (6, H, W) uint8 npz — one file per frame.
            np.savez_compressed(preds_dir / f"{name}.npz", binary_stack=binary_stack)

        if idx in overlay_indices:
            # 3-up per class would be busy; instead make a tiled (input + 6 binary)
            # overlay so the 12 sampled frames are skimmable.
            tile_h = target_h // 2
            tile_w = target_w // 2
            input_small = np.asarray(pil.resize((tile_w, tile_h)))
            tiles = [input_small]
            for c, name_c in enumerate(class_names):
                bin_small = np.asarray(
                    Image.fromarray((binary_stack[c] * 255).astype(np.uint8))
                    .resize((tile_w, tile_h), Image.NEAREST))
                tile_rgb = np.stack([bin_small] * 3, axis=-1)
                tiles.append(tile_rgb)
            row0 = np.concatenate(tiles[:4], axis=1)
            row1 = np.concatenate(tiles[4:] + [np.zeros_like(tiles[0])], axis=1)
            panel = np.concatenate([row0, row1], axis=0)
            Image.fromarray(panel).save(overlays_dir / f"{name}.png")

    elapsed = time.time() - t0
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1e-12), np.nan)
    miou = float(np.nanmean(iou))

    with open(out_dir / "results.txt", "w") as f:
        f.write(f"# OTAS eval — Table V paper protocol (Simon's PR #2 review comment)\n")
        f.write(f"protocol: 6-class binary per-class, neg=['thing'], threshold={args.threshold}\n")
        f.write(f"split: {args.split}\n")
        f.write(f"n_frames: {n_frames}\n")
        f.write(f"n_classes: {n_cls}\n")
        f.write(f"input_h: {args.input_h}\n")
        f.write(f"input_w: {args.input_w}\n")
        f.write(f"elapsed_seconds: {elapsed:.1f}\n")
        f.write(f"mIoU_6cls: {miou * 100:.4f}\n")
        f.write("\nper_class_IoU:\n")
        for c, name in enumerate(class_names):
            val = iou[c]
            val_str = "nan" if np.isnan(val) else f"{val * 100:.4f}"
            f.write(f"  {name} (raw id {raw_ids[c]}): {val_str}   "
                    f"tp={int(tp[c])} fp={int(fp[c])} fn={int(fn[c])}\n")
        f.write(f"\nthreshold: {args.threshold}\n")
        f.write(f"neg_prompts: {NEG_PROMPTS}\n")
        f.write(f"pos_prompts: {[n for (_, n) in PAPER_CLASS_PROMPTS]}\n")
        f.write(f"raw_id_to_prompt: {dict(PAPER_CLASS_PROMPTS)}\n")

    print(f"\n[own_RELLIS_paper] mIoU(6cls) = {miou * 100:.2f}%   "
          f"({n_frames} frames, {elapsed:.0f}s)")
    print("per-class IoU:")
    for c, name in enumerate(class_names):
        val_str = "nan" if np.isnan(iou[c]) else f"{iou[c] * 100:6.2f}%"
        print(f"  {name:<10s} (raw id {raw_ids[c]:>2d}): {val_str}")


if __name__ == "__main__":
    main()
