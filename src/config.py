import os, json, sys
from pathlib import Path

from torch.cuda import is_available as cuda_is_available
target_device = "cuda" if cuda_is_available() else "cpu"

#if os.isatty(sys.stdout.fileno()): current_dir = os.getcwd() # TODO: Behaviour in tty is unreliable due to SAM not being found
current_dir = os.path.dirname(os.path.abspath(__file__))

config = {
    ### VISION MODEL PARAMS
    "log_time": False,
    "enable_amp_autocast": True,
    
    ## DINOV2, Params are forwarded to "vit_small". See https://github.com/facebookresearch/dinov2/issues/316#issuecomment-1816378994
    "dinov2_params": {
        "patch_size": 14, "img_size": 518, "init_values": 1,
        "block_chunks": 0, "num_register_tokens": 4,
    },
    "dinov2_model_path": f"{current_dir}/foundation_models/dinov2_checkpoints/dinov2_vits14_reg4_pretrain.pth",
    "dinov2_device": target_device, # can be individually overwritten
       
    ## MASK CLIP
    "clip_device": target_device,
    "clip_model_type": "ViT-B/16",
    "clip_model_path": f"{current_dir}/foundation_models/clip_checkpoints/ViT-B-16.pt",

    ## SAM2
    "sam2_model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2_model_path": f"{current_dir}foundation_models/sam2_checkpoints/sam2.1_hiera_tiny.pt",
    "sam2_device": target_device,

    ## SPATIAL
    "enable_spatial": True, # Disable this to skip slow Open3D import
    "spatial_device": target_device, # Disable if you have little GPU memory. Embedding will still be done on GPU
    "spatial_max_depth": 100,

    ## MODEL CONFIG
    "n_clusters": 24, # Sensible defaults for complex scenes -> If the number of semantic classes is known, set n_clusters and n_components to that number
    "n_components": 48,
    "dino_scale_factor": 2,
    "shared_feat_resolution": 32,
    "dim_reduction_type": "pca_gpu",
    "cluster_model_type": "kmeans",
    "enable_mask_refinement": True,
    
    ## CAMERA CALIBRATION (Only for 3D)
    "cam0_params": {
        "fu_fv_cu_cv": [457.587, 456.134, 379.999, 255.238], # default from euroc cam1 (used for debugging). Set according to your camera for spatial to work properly
        "dist_coeffs": [-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05],
        "distortion_model": "radial-tangential"
    }
}

if "OTAS_CONFIG_PATH" in os.environ: # allows parts of the global config to be overridden
    config_path = Path(os.environ.get("OTAS_CONFIG_PATH", ""))
    print(f"[OTAS]: Using custom config from: {config_path}")
    assert config_path.is_file(), f"Config file not found: {config_path}"
    assert config_path.suffix == ".json", f"Expected a .json config file, got: {config_path.suffix}"

    with open(config_path, "r") as f: custom_config = json.load(f)
    config.update(custom_config)
