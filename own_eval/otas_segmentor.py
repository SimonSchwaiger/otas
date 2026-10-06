# Multi-class semantic-segmentation adapter for OTAS.
#
# Purpose:
#   OTAS's public API (`single_inference.similarity_single`) takes a list of `pos_prompts` and
#   `neg_prompts` and collapses them to a single per-pixel similarity map via
#   `mean(pos_sims) - mean(neg_sims)` (see src/model.py:semantic_mask.similarity). That shape
#   is binary-flavoured: one score map per (pos, neg) prompt set, then thresholded.
#
#   To compare OTAS against RADSeg / OpenRSS on the unified 6-dataset scoreboard we need
#   per-pixel N-way argmax across 5–22 classes (one of which is the unified `unknown` at
#   contig 0). This adapter bypasses the aggregation step and instead:
#
#       1. Runs OTAS's language_map once per image, producing a pooled
#          (shared_feat_resolution, shared_feat_resolution, 512) embedding map.
#       2. Encodes each of the N class names through MaskCLIP — using the BARE class string
#          only, no prompt template (no "a photo of …"), no negative prompts.
#       3. Calls OTAS's lower-level `clip_similarity` einsum per class, producing N low-res
#          (H_lr, W_lr) similarity maps.
#       4. Stacks them to (N, H_lr, W_lr), bilinear-upsamples to (N, H, W) (the same step
#          OTAS's `similarity_single` already uses for its single-map case), then argmaxes
#          across the class axis to get a (H, W) uint8 prediction tensor.
#
# Prompt-format rationale (user-specified for this evaluation):
#   - Bare class names, no template. RADSeg uses an imagenet-style 20-variant template,
#     OpenRSS uses `"a pohot of "` (sic). Each system gets the prompt format the user
#     specified for it; we do not apply templates that weren't part of OTAS's design.
#   - No negative prompts. Argmax across all N positive prompts (one of which is
#     `"unknown"` at contig 0) is the entire scoring rule.
#   - SAM2 mask refinement is off (matches RADSeg / OpenRSS which have no post-processing).
#
# Output contract:
#   `OTASEncoder.predict(pil_img)` returns `(preds, probs)` where
#       preds: np.uint8 (H, W)        per-pixel argmax class IDs in [0..N-1]
#       probs: np.float32 (H, W, N)   raw cosine similarities (NOT softmaxed; used only for
#                                     downstream introspection / saved overlays).
#   The output shape matches RADSeg's `(seg_probs, seg_preds)` ordering so per-dataset
#   eval scripts can mirror the RADSeg / OpenRSS scaffolding one-for-one.

import os
import sys
import json
import copy
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# OTAS's `src/` is not a package; add it to sys.path so `import model` resolves to OTAS's
# model.py, not torchvision.models or any other shadowing module.
_OTAS_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _OTAS_SRC not in sys.path:
    sys.path.insert(0, _OTAS_SRC)


_DEFAULT_CONFIG = {
    # SAM2 + Open3D / spatial reconstruction off — we only want the 2D semantic head.
    "enable_mask_refinement": False,
    "enable_spatial": False,
    # Model compilation is broken on some torch.compile + Blackwell combinations and
    # only matters for repeated calls inside a single process. Disable for stability.
    "enable_model_compilation": False,
    # OTAS paper Table V / supplementary §VII.A configuration. These four knobs match
    # the published RELLIS-3D backbone-ablation protocol verbatim:
    #   - dinov2_input_size=224 yields DINOv2's native 16×16 patch grid (14-px patches),
    #     which is then bilinear-interpolated up to the d=64 shared resolution.
    #   - shared_feat_resolution=64 = the d=64 shared feature grid used by the paper.
    #   - n_clusters=24, n_components=24 = k=24, Cr=24 from the paper.
    # These apply to every dataset on the scoreboard, not just RELLIS, so the OTAS
    # column reports numbers under a single internally consistent config.
    "dinov2_input_size": 224,
    "dino_scale_factor": 2,
    "shared_feat_resolution": 64,
    "n_clusters": 24,
    "n_components": 24,
    "enable_amp_autocast": True,
}


def _write_temp_config(custom_overrides: Optional[dict] = None) -> str:
    # OTAS picks up config overrides via the OTAS_CONFIG_PATH env var, which it parses as a
    # JSON file. Write our overrides to a tempfile and return the path; OTAS keys not present
    # here fall back to src/config.py defaults.
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    if custom_overrides:
        cfg.update(custom_overrides)
    cfg_path = Path("/tmp") / f"otas_adapter_cfg_{os.getpid()}.json"
    cfg_path.write_text(json.dumps(cfg))
    return str(cfg_path)


class OTASEncoder:
    # Instantiate once with the dataset's class list, then call .predict(pil_img) per frame.
    #
    # The class list must include the unified `"unknown"` token at index 0 (matching the
    # RADSeg / OpenRSS scoreboard convention). The rest are bare foreground class names from
    # the dataset's ontology (e.g. `"grass"`, `"trees"`, `"sky"`, …) — they are encoded with
    # CLIP's text encoder as-is, with no prompt template applied.
    def __init__(self, class_names: List[str], config_overrides: Optional[dict] = None):
        assert len(class_names) >= 2, "Need at least 2 classes for argmax to be meaningful."
        assert class_names[0].lower() == "unknown", (
            f"Class 0 must be 'unknown' to match the unified ignore-class convention; "
            f"got {class_names[0]!r}."
        )
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)

        cfg_path = _write_temp_config(config_overrides)
        # Lazy import: importing OTAS's model.py triggers heavy backbone loads (DINOv2,
        # MaskCLIP). We want that to happen exactly once, inside __init__, after the
        # OTAS_CONFIG_PATH env var is set.
        os.environ["OTAS_CONFIG_PATH"] = cfg_path
        import model  # noqa: E402
        import vision_utils  # noqa: E402

        self._config = copy.deepcopy(model.config)
        self._language_map = model.language_map(config=self._config)
        # Hold a reference to the global featurizer so we don't reload checkpoints.
        self._featurizer = vision_utils.featurizer
        self._clip_similarity = vision_utils.clip_similarity

        # SAM2 multi-class refinement reuses OTAS's reference path verbatim. OTAS's
        # `semantic_mask.binary_mask_refined` already encapsulates the full single-class
        # SAM2 pipeline (clip_similarity → normalize → threshold → upsample to 256 →
        # logits → sam2.predict → best-scored logits). To get an N-way argmax we call
        # `binary_mask_refined(..., ret_dict=True)` once per class with that class as the
        # sole positive prompt + `""` as the (no-op) negative prompt, pull the returned
        # `pred_logits`, then argmax across the per-class SAM-refined logits. This way
        # the SAM-on path leans on OTAS's tested code rather than re-implementing
        # threshold/upsample/clamp/predict logic in the adapter.
        self._enable_mask_refinement = bool(self._config.get("enable_mask_refinement", False))
        self._mask_instance = (
            model.semantic_mask(config=self._config)
            if self._enable_mask_refinement else None
        )

        # Pre-encode all class-name text embeddings once. Each is a (D,) float32 tensor on
        # the configured CLIP device. Bare class string, no prompt template.
        self._text_feats = []
        for name in self.class_names:
            feat = self._featurizer.clip_encode_text(name)["features"].detach().clone()
            self._text_feats.append(feat)
        # Stack to (N, D) for convenience; per-class einsum is still done in a loop because
        # `clip_similarity` is shaped for a single text vector at a time.
        self._text_feats_stack = torch.stack(self._text_feats, dim=0)  # (N, D)

        self._device = self._config["clip_device"]

    @torch.no_grad()
    def predict(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        # Returns (preds, probs):
        #   preds: np.uint8 (H, W)         argmax class IDs over the bare-name prompt set.
        #   probs: np.float32 (H, W, N)    raw cosine similarities at full resolution.
        # The probs tensor is mostly for debugging / saving raw heatmaps; downstream eval
        # only consumes preds.
        if img.mode != "RGB":
            img = img.convert("RGB")
        orig_h, orig_w = img.height, img.width

        # 1. Pooled DINOv2 + MaskCLIP embedding map: (H_lr, W_lr, D)
        pooled = self._language_map.embed_image(img)
        # 2. Reshape to (1, D, H_lr, W_lr) for `clip_similarity` which expects "chw" layout.
        img_feats = pooled.permute(2, 0, 1).unsqueeze(0)

        # 3. Per-class cosine similarity. Loop over N classes; each einsum is cheap relative
        # to the DINOv2 forward pass that already ran in step 1.
        per_class_sims = []
        for text_feat in self._text_feats:
            sim_lr = self._clip_similarity(img_feats, text_feat)  # (H_lr, W_lr)
            per_class_sims.append(sim_lr)
        sims_lr = torch.stack(per_class_sims, dim=0)  # (N, H_lr, W_lr)

        # 4. Upsample each class's similarity map to (H, W) via bilinear interp (same op
        # OTAS's `similarity_single` uses for its single-map case). Add a batch+channel
        # axis so F.interpolate's "bilinear" mode is well-defined.
        sims_up = F.interpolate(
            sims_lr.unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (N, H, W)

        # 5a. SAM2 multi-class refinement (only if enable_mask_refinement=True).
        # Delegate per-class refinement to OTAS's own `binary_mask_refined` so we reuse
        # the exact threshold / upsample / clamp / SAM2.predict / best-score pipeline
        # the OTAS authors maintain; we only add the N-class argmax on top.
        if self._enable_mask_refinement:
            sims_refined = self._refine_with_sam(img, pooled, orig_h, orig_w)
            preds = sims_refined.argmax(axis=0).astype(np.uint8)
            probs = sims_refined.transpose(1, 2, 0).astype(np.float32)  # (H, W, N)
            return preds, probs

        # 5b. No-SAM path — argmax across the class axis. Cast to uint8 — class IDs fit
        # (N <= 22 in our scoreboard) and uint8 PNGs are the canonical cached-pred
        # format on disk.
        preds = sims_up.argmax(dim=0).to(torch.uint8).cpu().numpy()
        probs = sims_up.permute(1, 2, 0).to(torch.float32).cpu().numpy()  # (H, W, N)
        return preds, probs

    def _refine_with_sam(self, img: Image.Image, pooled_features: torch.Tensor,
                         orig_h: int, orig_w: int) -> np.ndarray:
        # Apply OTAS's own `semantic_mask.binary_mask_refined` per class and stack the
        # SAM-returned logits into a (N, orig_h, orig_w) numpy float32 array suitable
        # for argmaxing.
        #
        # For each class c we call OTAS verbatim:
        #     self._mask_instance.binary_mask_refined(
        #         shared_feature_map=pooled_features,
        #         pos_prompts=[class_name],
        #         neg_prompts=[""],
        #         original_img=img,
        #         ret_dict=True,
        #     )
        # which internally runs clip_similarity → min-max normalise → threshold@0.5 →
        # bilinear upsample to 256x256 → scale to logits in [-9.9999, 9.9999] →
        # sam2_model.set_image → sam2_model.predict → pick best-scored output.
        # We pull `pred_logits` from the returned dict and upsample to full image
        # resolution so all classes' refined logits share a grid before argmax.
        #
        # The bare-class prompt protocol is preserved: pos = [class_name],
        # neg = [""] — OTAS's similarity() treats the [""] neg list as a zero map, so
        # this matches the no-SAM adapter's "score raw per-class similarity, argmax
        # across N classes" rule exactly.
        assert self._mask_instance is not None, (
            "SAM2 not loaded — instantiate OTASEncoder with "
            "config_overrides={'enable_mask_refinement': True}.")

        n_cls = len(self.class_names)
        refined = np.zeros((n_cls, orig_h, orig_w), dtype=np.float32)
        for c, name in enumerate(self.class_names):
            out = self._mask_instance.binary_mask_refined(
                shared_feature_map=pooled_features,
                pos_prompts=[name],
                neg_prompts=[""],
                original_img=img,
                ret_dict=True,
            )
            sam_logits = out["pred_logits"]  # SAM2's native low-res logits, (h, w)
            sam_logits_t = torch.from_numpy(sam_logits).unsqueeze(0).unsqueeze(0).float()
            sam_logits_up = F.interpolate(sam_logits_t, size=(orig_h, orig_w),
                                          mode="bilinear", align_corners=False)
            refined[c] = sam_logits_up.squeeze().numpy()
        return refined
