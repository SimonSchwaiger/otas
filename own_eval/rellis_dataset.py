# RELLIS-3D Dataset (Texas A&M off-road autonomous-driving benchmark) loader for OTAS.
#
# Yields `(image_f (4,H,W), label (H,W), name_safe (str))` tuples — the same shape the
# OTAS `own_eval/eval_common.run_eval` loop expects. Self-contained so the RELLIS-3D
# replication can be reproduced from a single OTAS checkout, no sibling repos required.
# Mirrors `/home/ubuntu/code/OpenRSS/util/RELLIS_dataset.py` one-for-one — keep them in
# sync if you change one.
#
# RELLIS-3D ships:
#   - RGB:  pylon_camera_node/<seq>/<stem>.jpg            (1920x1200)
#   - Label-id PNG: pylon_camera_node_label_id/<seq>/<stem>.png  (1920x1200, uint8)
#   - NO thermal channel (sensors are LiDAR + Basler RGB + Nerian stereo + VN-300 INS).
#
# The dataset root also contains `train.lst` / `val.lst` / `test.lst`. Each line is two
# whitespace-separated paths relative to the dataset root: `<rel_image_path> <rel_label_path>`.
# (The legacy HRNet loader has a 1-column branch for test mode, but the released test.lst
# is actually 2-col like train/val — confirmed by inspection.)
#
# Class IDs in the RELLIS ontology are sparse: {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17,
# 18, 19, 23, 27, 31, 33, 34}. We remap to contiguous [0..19] under the unified
# unknown-incl convention: raw 0 (void) -> contig 0 (unknown); raw {1, 3, 4, ..., 34} ->
# contig {1, 2, 3, ..., 19} in the order documented by Rellis-3D/ontology.yaml; any other
# raw ID (gap values 2/11/13/14/16/20-22/24-26/28-30/32/35/36) folds to contig 0
# (unknown) via the 256-entry LUT default. This is byte-identical to what
# `RemapRellisUnknownLabel` does in RADSeg/evaluation/2d/custom_datasets.py — same
# mapping, same class order.
#
# Modality contract: RELLIS has only one valid eval mode (RGB). There is no `mask_modality`
# argument and no thermal channel — earlier revisions accepted a `mask_modality` flag for
# "API parity with GOD_dataset", but the four-way enum (`none`/`rgb_only`/`thermal_only`)
# was a semantic lie on RELLIS: `none` = "RGB + zero-padded T" (not RGB+T fusion); `rgb_only`
# was a no-op (the T was already zero); `thermal_only` silently produced an all-zero input
# and ran garbage IoU. No caller exercised any mode other than `none`, so the parameter has
# been removed.
#
# Returned tuple: `(image_f (4,H,W) float[0,1] R G B 0, label (H,W) int64 in 0..19,
# name_safe (str))`. The 4th channel is constant zero (no thermal sensor) — kept on the
# tensor only so the same SAM 4-channel input convention works.

import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import PIL
from PIL import Image


# Order matches Rellis-3D/ontology.yaml + RADSeg's RELLIS_UNKNOWN_CLASSES. Singular `tree`
# is intentional — kept verbatim from the source ontology (vs GOD's plural `trees`).
RELLIS_CLASS_NAMES = [
    "unknown", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle",
    "object", "asphalt", "building", "log", "person", "fence", "bush",
    "concrete", "barrier", "puddle", "mud", "rubble",
]
RELLIS_NUM_CLASSES = len(RELLIS_CLASS_NAMES)  # 20 incl unknown

# Raw RELLIS GT id -> contig id under the unknown-incl convention.
# Raw 0 (void) -> 0 (unknown). 19 fg raw ids -> contig 1..19 in ontology-yaml order.
RELLIS_RAW_TO_CONTIGUOUS = {
    0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    12: 10, 15: 11, 17: 12, 18: 13, 19: 14, 23: 15, 27: 16, 31: 17,
    33: 18, 34: 19,
}


def _build_remap_lut() -> np.ndarray:
    # 256-entry LUT, default 0 (unknown). Gap raw IDs (2, 11, 13, 14, 16, 20-22, 24-26,
    # 28-30, 32, 35, 36) silently fold to unknown — matches RADSeg's LUT semantics.
    lut = np.zeros(256, dtype=np.uint8)
    for raw, mapped in RELLIS_RAW_TO_CONTIGUOUS.items():
        lut[raw] = mapped
    return lut


_REMAP_LUT = _build_remap_lut()


class RELLIS_dataset(Dataset):
    # Yields (image_tensor (4,H,W) float[0,1] with channel 3 = constant zero, label_tensor
    # (H,W) int64, name_safe (str)) so OTAS's eval_common.run_eval can consume it unchanged.
    #
    # `data_dir` should point at the extracted Rellis-3D root (the directory containing the
    # split lists and the 5 sequence subdirs 00000..00004). The split list is read directly
    # from `<data_dir>/<split>.lst` — no separate `split_dir` is needed; RELLIS ships them
    # at the dataset root.
    def __init__(self, data_dir, split="test", split_dir=None, input_h=None, input_w=None,
                 transform=None):
        # input_h/input_w default to None — when unset the Basler pylon JPGs
        # and matching label PNGs (both 1920×1200 natively) are returned at
        # their native shape with no resize. Pass explicit (input_h, input_w)
        # to score on a custom grid.
        super().__init__()
        self.data_dir = data_dir
        # split_dir is accepted for API parity with GOD_dataset but ignored — RELLIS ships
        # the split lists at the dataset root, not a sibling directory.
        self.split_dir = split_dir or data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform or []
        self._lut = _REMAP_LUT
        self.class_names = RELLIS_CLASS_NAMES

        # Parse the split file. Each line has two whitespace-separated relative paths.
        split_path = os.path.join(self.split_dir, split + ".lst")
        self.pairs = []  # list of (img_rel, label_rel)
        with open(split_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # The released test.lst has 2 cols like train/val (the upstream HRNet
                # loader's 1-col test branch is legacy / unused) — fail loudly if that
                # assumption is ever violated by a future RELLIS release.
                assert len(parts) == 2, (
                    f"RELLIS split line must have 2 paths, got {len(parts)}: {line!r}")
                self.pairs.append((parts[0], parts[1]))
        self.n_data = len(self.pairs)

    def __getitem__(self, index):
        img_rel, label_rel = self.pairs[index]
        # name_safe: replace path separators with '_' so eval_common can use it as a single
        # filename component. Strip the .jpg extension. e.g. "00000_pylon_camera_node_frame000000-1581624652_750"
        stem, _ext = os.path.splitext(img_rel)
        name_safe = stem.replace("/", "_")

        # RGB at 1920x1200; resize to (input_h, input_w) if those are set, else
        # keep native.
        rgb_path = os.path.join(self.data_dir, img_rel)
        rgb = np.asarray(PIL.Image.open(rgb_path).convert("RGB"))  # (1200, 1920, 3) uint8

        # Label from pylon_camera_node_label_id (same resolution as RGB).
        # Confirmed across the released test split: every label PNG is PIL mode "L",
        # (1200, 1920) uint8, holding raw class IDs in the documented sparse set
        # {0, 1, 3-10, 12, 15, 17-19, 23, 27, 31, 33-34}. Colored annotations live
        # in the sibling pylon_camera_node_label_color/ directory and are never
        # loaded here. Remap via LUT to contig [0..19].
        label_path = os.path.join(self.data_dir, label_rel)
        raw_label = np.asarray(PIL.Image.open(label_path))  # (1200, 1920) uint8
        remapped = self._lut[raw_label]

        # input_h/input_w None ⇒ keep native (RGB and label are both 1920×1200
        # so this is a true no-op); else bilinear-resize RGB, NEAREST-resize
        # label.
        if self.input_h is None and self.input_w is None:
            rgb_resized = rgb
            label = remapped.astype(np.int64)
            th_h, th_w = rgb.shape[:2]
        else:
            rgb_resized = np.asarray(
                PIL.Image.fromarray(rgb).resize((self.input_w, self.input_h),
                                                resample=PIL.Image.BILINEAR)
            )
            label = np.asarray(
                PIL.Image.fromarray(remapped).resize((self.input_w, self.input_h),
                                                     resample=PIL.Image.NEAREST),
                dtype=np.int64,
            )
            th_h, th_w = self.input_h, self.input_w

        # 4th channel is constant zero — RELLIS-3D has no thermal sensor. Kept on the tensor
        # so the same SAM 4-channel input convention works; downstream modality switches in
        # GOD-style eval loops are not relevant here.
        th_resized = np.zeros((th_h, th_w), dtype=np.uint8)

        # (H, W, 4) uint8 R G B 0
        image = np.dstack([rgb_resized, th_resized])

        for func in self.transform:
            image, label = func(image, label)

        # `image` is already (input_h, input_w, 4) uint8 — RGB was resized at
        # load time and dstack'd with the zero thermal channel. So we only need
        # the uint8 -> float32/255 cast and the HWC -> CHW transpose that
        # PyTorch expects. The earlier PIL round-trip-with-resize here was a
        # copy-paste from MF_dataset (where it IS load-bearing because that
        # adapter resizes at this final step) and ran an identity resize on
        # every frame.
        image_f = image.astype(np.float32).transpose(2, 0, 1) / 255

        return torch.tensor(image_f), torch.tensor(label), name_safe

    def __len__(self):
        return self.n_data
