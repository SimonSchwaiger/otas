# OTAS open-vocabulary evaluation on RELLIS-3D (Texas A&M off-road autonomous-driving dataset).
#
# Mirrors own_GOD.py one-for-one, with two differences:
#   1. RGB-only — RELLIS-3D ships no thermal modality (sensors: LiDAR + Basler RGB + Nerian
#      stereo + VN-300 INS). The 4-channel image tensor still gets emitted by RELLIS_dataset
#      for API parity, but channel 3 is all-zero and we never run modality="thermal" here.
#   2. SAM ablation — accepts --enable_mask_refinement to toggle OTAS's SAM2 mask refinement
#      head on top of the bare DINOv2 + MaskCLIP token alignment. Default off (matches the
#      paper's Table V row "OTAS w. DINOv2 ViT-S/14, raw class labels, no mask refinement").
#
# Output dir convention:
#   result/Pred/RELLIS_rgb/      <- SAM off (paper Table V replication target = 48.48 mIoU)
#   result/Pred/RELLIS_rgb_sam/  <- SAM on  (new ablation, not in the paper)
#
# Two invocations are needed to produce both numbers (a single process can't easily flip
# the SAM toggle since OTAS's config is locked at OTASEncoder construction time).

import argparse
import sys
from pathlib import Path

# The RELLIS_dataset.py is vendored next to this script (own_eval/rellis_dataset.py) so
# the RELLIS-3D replication is fully self-contained inside an OTAS checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # eval_common, otas_segmentor, rellis_dataset

from rellis_dataset import RELLIS_dataset, RELLIS_CLASS_NAMES
from eval_common import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ubuntu/mnt/rellis_3d/Rellis-3D",
                        help="Rellis-3D root dir (contains train.lst/val.lst/test.lst and 00000..00004/).")
    parser.add_argument("--split", default="test",
                        help="One of train|val|test (each .lst file has 2 cols: <img> <label>).")
    parser.add_argument("--out_root", default="/home/ubuntu/code/OTAS/result/Pred")
    # input_h/input_w default to None → RELLIS_dataset returns frames at native
    # 1920×1200 with no resize. DINOv2 always resizes internally to
    # dinov2_input_size (224), so the dataset H/W only controls the GT scoring
    # grid and the bilinear pred-upsample target — native is the right answer
    # for DINOv2. The paper §VII.A wording of "1024×1024" was specifically
    # about getting AM-RADIO/DINOv3's 16-pixel patches to land exactly on the
    # d=64 shared grid, not about DINOv2 (which the OTAS repo ships as
    # default). Pass explicit --input_h --input_w to override.
    parser.add_argument("--input_h", type=int, default=None)
    parser.add_argument("--input_w", type=int, default=None)
    parser.add_argument("--enable_mask_refinement", action="store_true",
                        help="Turn OTAS SAM2 mask refinement on. Paper Table V uses "
                             "no mask refinement; this flag is for ablation only.")
    parser.add_argument("--out_suffix", default=None,
                        help="Override the output dir suffix. Default: '_sam' if SAM on, "
                             "empty otherwise. Set explicitly to override.")
    args = parser.parse_args()

    if args.out_suffix is None:
        suffix = "_sam" if args.enable_mask_refinement else ""
    else:
        suffix = args.out_suffix
    out_dir = Path(args.out_root) / f"RELLIS_rgb{suffix}"

    input_h, input_w = args.input_h, args.input_w

    dataset = RELLIS_dataset(
        data_dir=args.data_dir,
        split=args.split,
        input_h=input_h,
        input_w=input_w,
    )
    class_names = RELLIS_CLASS_NAMES

    # Smoke-check: make sure the dataset loaded sane label values before paying the
    # 10-minute encoder warmup + ~1672-frame eval cost. Same invariant as RADSeg's
    # verify_rellis.py "(3) GT remap on real frames" check.
    print(f"[own_RELLIS] {len(class_names)} classes; "
          f"dataset size: {len(dataset)} frames; sample IDs in 3 frames:")
    import numpy as np
    valid = set(range(len(class_names)))
    seen = set()
    for i in (0, len(dataset) // 2, len(dataset) - 1):
        _, lbl, name = dataset[i]
        ids = set(np.unique(lbl.numpy()).tolist())
        leaks = ids - valid
        assert not leaks, f"frame {i} ({name}) leaked GT values: {sorted(leaks)}"
        seen |= ids
        print(f"  frame {i:>4} ({name}): GT ids = {sorted(ids)}")
    print(f"[own_RELLIS] union of GT ids over 3 frames: {sorted(seen)} (all in 0..{len(class_names)-1})\n")

    # OTAS paper Table V / §VII.A defaults — shared_feat_resolution=64, n_components=24,
    # dinov2_input_size=224, k=24, no mask refinement — are now baked into
    # otas_segmentor.py:_DEFAULT_CONFIG, so this driver doesn't need to override them.
    config_overrides = {"enable_mask_refinement": bool(args.enable_mask_refinement)}
    run_eval(
        dataset=dataset,
        class_names=class_names,
        out_dir=str(out_dir),
        modality="rgb",
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()
