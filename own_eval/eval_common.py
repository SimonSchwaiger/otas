# Shared evaluation loop for OTAS on the unified 6-dataset scoreboard.
#
# Provides one entry point — `run_eval(dataset, class_names, out_dir, modality)` — that
# every per-dataset script (own_GOD.py, own_BASEPROD.py, …) wraps with the dataset-
# specific class list and DataLoader. The point of pulling this out is so the modality
# handling, OTAS inference, prediction caching, and IoU metric computation are written
# once and identical across all six datasets — matching the convention RADSeg's
# eval.py and OpenRSS's own_*.py use, except deduplicated.
#
# Modality contract:
#   The OpenRSS datasets we re-use yield `(image_f, label, name)` where
#   `image_f` is a (4, H, W) float tensor in [0, 1] with channels [R, G, B, T].
#   For OTAS (which is RGB-trained DINOv2 + MaskCLIP) we expose two modalities:
#     - "rgb":     pass channels [R, G, B] through OTAS unchanged.
#     - "thermal": replicate channel T to 3 channels (R = G = B = T) — mirrors how
#                  RADSeg's "thermal" column is generated. DINOv2 is out-of-distribution
#                  on thermal, which is the whole point of the comparison.
#   An earlier revision also emitted a 4th `th_vis` tensor (jet-colormapped thermal)
#   for qualitative panels; it was never consumed and has been removed from every
#   adapter to save the per-frame colormap+resize cost.
#
# Output layout (per dataset × modality):
#   <out_dir>/preds/<name>.png            uint8 grayscale, per-pixel class IDs.
#   <out_dir>/results.txt                 full + fg-only mIoU and per-class IoU.
#   <out_dir>/overlays/<sampled_names>.png   3-up [input | gt | pred] panels for a
#                                            sparse sample (visual sanity only).

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sibling otas_segmentor importable when invoked as `python -m own_eval.own_GOD`
# from the OTAS root, or as `python own_eval/own_GOD.py` directly.
_OWN_EVAL_DIR = str(Path(__file__).resolve().parent)
if _OWN_EVAL_DIR not in sys.path:
    sys.path.insert(0, _OWN_EVAL_DIR)


def _tensor_to_pil_rgb(image_f: torch.Tensor) -> Image.Image:
    # image_f: (4, H, W) float in [0, 1] from the OpenRSS DataLoader.
    # Returns a PIL RGB image of the first 3 (RGB) channels.
    rgb = image_f[:3].numpy()
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(rgb_u8, mode="RGB")


def _tensor_to_pil_thermal_as_rgb(image_f: torch.Tensor) -> Image.Image:
    # Replicate the thermal channel (image_f[3]) to a 3-channel PIL RGB so DINOv2's
    # ImageNet-mean normalization sees plausible-ish R = G = B input. This matches
    # RADSeg's "thermal" treatment exactly — same image stream, different model
    # encoder.
    th = image_f[3].numpy()
    th_u8 = (np.clip(th, 0.0, 1.0) * 255.0).astype(np.uint8)
    th_rgb = np.stack([th_u8, th_u8, th_u8], axis=-1)
    return Image.fromarray(th_rgb, mode="RGB")


def _iou_per_class(conf: np.ndarray) -> np.ndarray:
    # Symmetric IoU = TP / (TP + FP + FN) per class, derived from a (N, N) confusion
    # matrix where rows are ground-truth and columns are predictions. Returns NaN for
    # any class that has zero presence in both gt and prediction (so it doesn't drag
    # the mean down to zero artificially).
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1e-12), np.nan)
    return iou


def _resolve_palette(n_classes: int, palette: Optional[List[List[int]]]) -> np.ndarray:
    # Returns an (>=n_classes, 3) uint8 LUT. If `palette` is given, it must have
    # at least n_classes rows in the dataset's class order (row i = color for
    # class i); otherwise we fall back to matplotlib tab20. Hand-curated palettes
    # let GOD/BASEPROD overlays color-match the published ontology keys
    # (and the RADSeg viz grid), instead of getting an arbitrary tab20 mapping.
    if palette is not None:
        arr = np.asarray(palette, dtype=np.uint8)
        assert arr.ndim == 2 and arr.shape[1] == 3, f"palette must be (N, 3) RGB, got {arr.shape}"
        assert arr.shape[0] >= n_classes, (
            f"palette has {arr.shape[0]} colors but dataset has {n_classes} classes"
        )
        return arr
    from matplotlib import cm
    cmap = cm.get_cmap("tab20", max(n_classes, 20))
    return (np.array([cmap(i)[:3] for i in range(max(n_classes, 20))]) * 255).astype(np.uint8)


def _save_palette_overlay(pred: np.ndarray, n_classes: int, out_path: Path,
                          palette: Optional[List[List[int]]] = None):
    lut = _resolve_palette(n_classes, palette)
    overlay = lut[pred]  # (H, W, 3)
    Image.fromarray(overlay).save(out_path)


def _save_3up(rgb_pil: Image.Image, gt: np.ndarray, pred: np.ndarray,
              n_classes: int, out_path: Path,
              palette: Optional[List[List[int]]] = None):
    # Side-by-side: input RGB | GT palette | Pred palette. Used for a sparse sample
    # of frames so we can eyeball alignment / failure modes without needing all
    # overlays on disk.
    lut = _resolve_palette(n_classes, palette)
    h, w = pred.shape
    rgb_np = np.asarray(rgb_pil.resize((w, h)))
    # Treat any GT id >= n_classes (e.g. 255 ignore) as a black pixel rather than
    # crashing the palette lookup.
    gt_safe = np.where(gt < n_classes, gt, 0).astype(np.uint8)
    gt_paint = lut[gt_safe]
    pred_paint = lut[pred]
    panel = np.concatenate([rgb_np, gt_paint, pred_paint], axis=1)
    Image.fromarray(panel).save(out_path)


def run_eval(
    dataset,
    class_names: List[str],
    out_dir: str,
    modality: str,
    *,
    num_overlay_samples: int = 12,
    ignore_label: int = 255,
    config_overrides: Optional[dict] = None,
    palette: Optional[List[List[int]]] = None,
    redraw_overlays_only: bool = False,
):
    # Args:
    #   dataset:      a torch Dataset that yields (image_f (4,H,W), label (H,W),
    #                 name (str)). The OpenRSS dataset adapters satisfy this
    #                 contract.
    #   class_names:  list of N strings. class_names[0] MUST be "unknown".
    #   out_dir:      where to write preds/, overlays/, results.txt.
    #   modality:     "rgb" or "thermal" (= thermal-as-RGB replica).
    #   num_overlay_samples: how many frames to emit 3-up overlays for. The full
    #                        preds/ dir always has every frame, but overlays are
    #                        sparse so we don't drown the disk.
    #   ignore_label: GT pixel value to treat as ignore (not counted in IoU). 255
    #                 by default — matches the OpenRSS dataset convention.
    #   config_overrides: optional dict of OTAS config keys to override (e.g.
    #                 {"enable_mask_refinement": True} to turn SAM2 on). Forwarded
    #                 verbatim to OTASEncoder; see otas_segmentor._DEFAULT_CONFIG.
    #   palette:      optional (N, 3) per-class RGB LUT in dataset class order.
    #                 If omitted, overlays use matplotlib tab20. Pass GOD's
    #                 GREAT_OUTDOORS_UNKNOWN_PALETTE to color-match the RADSeg
    #                 viz grid and the published GOD ontology key.
    #   redraw_overlays_only: skip OTAS inference and metric computation; only
    #                 re-render the sampled 3-up overlays from cached preds in
    #                 <out_dir>/preds/. Used to refresh visualizations after a
    #                 palette change without redoing the (slow) forward pass.
    #                 Requires preds/ to already exist on disk.
    assert modality in {"rgb", "thermal"}, f"modality must be rgb|thermal, got {modality!r}"

    # Import the encoder here so OTAS only loads once per process (and not at module
    # import time, which would prevent us from setting env vars first).
    from otas_segmentor import OTASEncoder

    out_dir = Path(out_dir)
    preds_dir = out_dir / "preds"
    overlays_dir = out_dir / "overlays"
    preds_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)

    # We deliberately keep batch_size=1 because OTAS's language_map operates on PIL
    # images one at a time (DINOv2 forward is autograd-disabled but not batched in
    # OTAS's reference path).
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    n_frames = len(dataset)

    # Pick `num_overlay_samples` evenly-spaced indices for the qualitative panels.
    overlay_indices = set(np.linspace(0, n_frames - 1, num=num_overlay_samples, dtype=int).tolist())

    chosen = (
        _tensor_to_pil_rgb if modality == "rgb" else _tensor_to_pil_thermal_as_rgb
    )

    # Redraw-only fast path: skip OTAS, skip metrics, just re-render overlays
    # from cached preds. Saves the ~5-min forward pass when we only want to
    # refresh visualizations after a palette change.
    if redraw_overlays_only:
        redrawn = 0
        for idx, batch in enumerate(loader):
            if idx not in overlay_indices:
                continue
            image_f, label, name = batch
            image_f = image_f.squeeze(0)
            label_np = label.squeeze(0).numpy()
            if isinstance(name, (list, tuple)):
                name = name[0]
            name = str(name)
            pred_path = preds_dir / f"{name}.png"
            if not pred_path.exists():
                print(f"[redraw] missing cached pred {pred_path}, skipping")
                continue
            pred_pil = Image.open(pred_path)
            # Cached preds may live at a different resolution than the current
            # dataset grid (e.g. an earlier run cached at pylon-native 1080×1440;
            # the current default is 480×640). Resize with NEAREST so the panel
            # composes — preds remain the canonical authoritative cache on disk.
            target_h, target_w = label_np.shape
            if pred_pil.size != (target_w, target_h):
                pred_pil = pred_pil.resize((target_w, target_h), Image.NEAREST)
            preds = np.asarray(pred_pil, dtype=np.uint8)
            pil = chosen(image_f)
            _save_3up(pil, label_np, preds, n_classes, overlays_dir / f"{name}.png",
                      palette=palette)
            redrawn += 1
        print(f"[{out_dir.name}] redrew {redrawn} overlays (modality={modality})")
        return

    encoder = OTASEncoder(class_names=class_names, config_overrides=config_overrides)

    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    t0 = time.time()

    pbar = tqdm(loader, desc=f"OTAS[{modality}] {out_dir.name}", total=n_frames)
    for idx, batch in enumerate(pbar):
        image_f, label, name = batch
        # DataLoader collates to a leading batch dim of 1 — strip it.
        image_f = image_f.squeeze(0)        # (4, H, W) float
        label_np = label.squeeze(0).numpy()  # (H, W) int64
        # `name` may be a list (DataLoader collates strings) — pull out the scalar.
        if isinstance(name, (list, tuple)):
            name = name[0]
        name = str(name)

        pil = chosen(image_f)
        preds, _probs = encoder.predict(pil)  # uint8 (H, W)

        # Bucket into the conf matrix, excluding ignore-label pixels.
        valid = label_np != ignore_label
        if valid.any():
            gt_valid = label_np[valid]
            pred_valid = preds[valid]
            # Clip just in case — preds are already in [0..n_classes-1] but defensive.
            gt_valid = np.clip(gt_valid, 0, n_classes - 1)
            pred_valid = np.clip(pred_valid, 0, n_classes - 1)
            bin_idx = gt_valid * n_classes + pred_valid
            counts = np.bincount(bin_idx, minlength=n_classes * n_classes)
            conf += counts.reshape(n_classes, n_classes)

        # Cache pred PNG (uint8 grayscale) — argmax IDs, no palette, so any
        # downstream tool can re-paint with its own palette.
        Image.fromarray(preds).save(preds_dir / f"{name}.png")

        if idx in overlay_indices:
            _save_3up(pil, label_np, preds, n_classes, overlays_dir / f"{name}.png",
                      palette=palette)

    elapsed = time.time() - t0
    iou = _iou_per_class(conf)
    full_miou = np.nanmean(iou)
    fg_miou = np.nanmean(iou[1:])  # excludes the 'unknown' class at contig 0

    # Write a single results.txt that's grep-friendly for the scoreboard updater.
    with open(out_dir / "results.txt", "w") as f:
        f.write(f"# OTAS eval — {out_dir.name}\n")
        f.write(f"modality: {modality}\n")
        f.write(f"n_frames: {n_frames}\n")
        f.write(f"n_classes: {n_classes}\n")
        f.write(f"elapsed_seconds: {elapsed:.1f}\n")
        f.write(f"full_mIoU: {full_miou * 100:.4f}\n")
        f.write(f"fg_only_mIoU: {fg_miou * 100:.4f}\n")
        f.write("\nper_class_IoU:\n")
        for name, val in zip(class_names, iou):
            val_str = "nan" if np.isnan(val) else f"{val * 100:.4f}"
            f.write(f"  {name}: {val_str}\n")
        f.write("\nconfusion_matrix_rows_gt_cols_pred:\n")
        for row in conf:
            f.write("  " + " ".join(str(int(c)) for c in row) + "\n")

    print(f"[{out_dir.name}] full mIoU = {full_miou * 100:.2f}%, "
          f"fg-only mIoU = {fg_miou * 100:.2f}% "
          f"({n_frames} frames, {elapsed:.0f}s)")
    return {"full_mIoU": full_miou, "fg_only_mIoU": fg_miou, "iou_per_class": iou}
