from config import config ## Must import config before this file. TODO: Assert config exists
import sys, inspect, os
from checking_and_logging import *

import math
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

import cv2
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from foundation_models.dinov2.vision_transformer import vit_small, vit_base, DinoVisionTransformer # This is unmodified from the original repo
from foundation_models.maskclip_onnx import clip # This is unmodified from the original repo

import contextlib
from tqdm import tqdm

# ====================
## Setup DINO V2
# Check provided config. If correct, instantiate model
function_signature = inspect.signature(DinoVisionTransformer)
dinov2_model = check_and_call(vit_small, config["dinov2_params"], function_signature) # Only instantiate dinov2 model if kwargs are accepted
assert config["dinov2_device"] in TORCH_DEVICES, f"Invalid torch device specified" # Only allowed torch devices
assert Path(config["dinov2_model_path"]).is_file(), f"Dino model path does not exist." # Model must exist and be of pth type
assert Path(config["dinov2_model_path"]).suffix == ".pth", f"Dino model is not of type '.pth'."
checkpoint = torch.load(config["dinov2_model_path"], map_location=config["dinov2_device"], weights_only=True)
dinov2_model.load_state_dict(checkpoint, strict=False)
_ = dinov2_model.eval().to(config["dinov2_device"])  # Set the model to evaluation mode if not training

# Helper function to only autocast if we explicitly enable it
def dino_maybe_autocast(enabled: bool = config["enable_amp_autocast"], device: str = config["dinov2_device"]): 
    return torch.amp.autocast(device) if enabled else contextlib.nullcontext()

## DinoV2 Preprocessing and basic inference
t_dino = transforms.Compose([transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

@torch.no_grad()
def dinov2_pipe(img: Image) -> Dict[str, Any]: # Convenience function to be consistent with huggingface pipeline
    with dino_maybe_autocast():
        img_tensor = t_dino(img).unsqueeze(0).to(config["dinov2_device"])
        with torch.no_grad(): x = dinov2_model.forward_features(img_tensor)
        return x
    
debug("Dinov2 pipeline instantiated")

# ====================
## Setup MASK CLIP
assert config["clip_device"] in TORCH_DEVICES, f"Invalid torch device specified" # Only allowed torch devices
if Path(config["clip_model_path"]).is_file():
    assert Path(config["clip_model_path"]).suffix == ".pt", f"CLIP model is not of type '.pt'."
    clip_name = config["clip_model_path"]
else: clip_name = config["clip_model_type"] ## Set to model type for download

clip_input_size = 224
t_clip = transforms.Compose([
    transforms.Resize(clip_input_size),
    # T.CenterCrop((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MaskCLIPFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load(
            clip_name, # "ViT-B/16",
            download_root="/src/foundation_models/clip_checkpoints"
        )
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        b, _, input_size_h, input_size_w = img.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

clip_model = MaskCLIPFeaturizer().to(config["clip_device"])

# Helper function to only autocast if we explicitly enable it
def clip_maybe_autocast(enabled: bool = config["enable_amp_autocast"], device: str = config["clip_device"]): 
    return torch.amp.autocast(device) if enabled else contextlib.nullcontext()

@torch.no_grad()
def clip_pipe(img: Image) -> Dict[str, Any]: # Convenience function to be consistent with huggingface pipeline
    with clip_maybe_autocast():
        img_tensor = t_clip(img).unsqueeze(0).to(config["clip_device"])
        img_features = clip_model(img_tensor)
        return {
            "features": img_features,
        }

@torch.no_grad()
def clip_encode_text(text_query: str) -> Dict[str, Any]:
    with clip_maybe_autocast():
        text = clip.tokenize(text_query).to(config["clip_device"])
        text_feats = clip_model.model.encode_text(text).squeeze().to(torch.float32)
        assert len(text_feats.shape) == 1
        return {
            "features": text_feats,
        }

# See: https://github.com/RogerQi/maskclip_onnx/blob/main/clip_playground.ipynb
@torch.no_grad()
def clip_similarity(img_features: torch.Tensor, text_feats: torch.Tensor) -> Dict[str, Any]:
    return torch.einsum("chw,c->hw", F.normalize(img_features[0].to(torch.float32), dim=0), F.normalize(text_feats, dim=0))

@torch.no_grad()
def clip_similarity_flat(img_features_flat: torch.Tensor, text_feats: torch.Tensor) -> Dict[str, Any]:
    return torch.einsum("nc,c->n", F.normalize(img_features_flat, dim=1), F.normalize(text_feats, dim=0))

debug("MaskCLIP pipeline instantiated")

# ====================
## Setup SAM2

if config["enable_mask_refinement"]:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    assert config["sam2_device"] in TORCH_DEVICES, f"Invalid torch device specified"
    assert Path(config["sam2_model_path"]).is_file(), "SAM2 model config does not exist."
    assert Path(config["sam2_model_path"]).suffix == ".pt", "SAM2 model config is not a .pt file."
    
    sam2_model = SAM2ImagePredictor(
        build_sam2(config["sam2_model_config"], config["sam2_model_path"], device=config["sam2_device"]) )

# ====================
## Setup IMAGE CALIBRATION -> Not implemented as function for more flexibility with multi cam
# class image_pipe_class:
#     def __init__(self, camera_config: Dict[str, Any]) -> None:
#         check_camera(camera_config["fu_fv_cu_cv"], camera_config["dist_coeffs"], camera_config["distortion_model"])
#         fu, fv, cu, cv = camera_config["fu_fv_cu_cv"]
#         self.camera_matrix = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], dtype=np.float32)
#         self.dist_coeffs = np.array(camera_config["dist_coeffs"])
#         self.distortion_model = camera_config["distortion_model"]
    
#     def __call__(self, img: Image) -> Dict[str, Any]:
#         img_array = np.array(img)
#         h, w = img_array.shape[:2]
#         new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 0)
#         rectified_img_array = cv2.undistort(img_array, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
#         fu, fv = new_camera_matrix[0, 0], new_camera_matrix[1, 1]
#         cu, cv = new_camera_matrix[0, 2], new_camera_matrix[1, 2]
#         return {
#             "image": Image.fromarray(rectified_img_array),
#             "fu_fv_cu_cv": np.array([fu, fv, cu, cv]),
#             "dist_coeffs": self.dist_coeffs
#         }

if config["enable_spatial"]:
    import open3d as o3d
    from scipy.spatial.transform import Rotation as R
    #calib_pipe = image_pipe_class(config["cam0_params"]) #TODO: conditionally load only if need to rectify
    debug("Camera calibration pipeline instantiated")
