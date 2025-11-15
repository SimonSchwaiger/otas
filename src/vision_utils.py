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

class OTASFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()

        ## Autocasting and dtype setup
        if config["clip_device"] == "cuda": self._clip_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else: self._clip_dtype = torch.float16

        if config["dinov2_device"] == "cuda": self._dino_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else: self._dino_dtype = torch.float16

        self._clip_maybe_autocast = (
            lambda: torch.amp.autocast(device_type=config["clip_device"], dtype=self._clip_dtype)
            if config["enable_amp_autocast"] else contextlib.nullcontext() )

        self._dino_maybe_autocast = (
            lambda: torch.amp.autocast(device_type=config["dinov2_device"], dtype=self._dino_dtype)
            if config["enable_amp_autocast"] else contextlib.nullcontext() )

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
        self.dinov2_model = dinov2_model.eval().to(config["dinov2_device"])
        dinov2_input_size = int(config["dinov2_input_size"])
        assert dinov2_input_size % config["dinov2_params"]["patch_size"] == 0, f"Dinov2 input size must be divisible by patch size"

        self.t_dino = transforms.Compose([transforms.Resize((dinov2_input_size, dinov2_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        debug("Dinov2 pipeline instantiated")

        # ====================
        ## Setup MASK CLIP
        clip_input_size = 224
        self.t_clip = transforms.Compose([
            transforms.Resize(clip_input_size),
            # T.CenterCrop((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])

        assert config["clip_device"] in TORCH_DEVICES, f"Invalid torch device specified" # Only allowed torch devices
        if Path(config["clip_model_path"]).is_file():
            assert Path(config["clip_model_path"]).suffix == ".pt", f"CLIP model is not of type '.pt'."
            clip_name = config["clip_model_path"]
        else: clip_name = config["clip_model_type"] ## Set to model type for download

        self.clip_model, self.clip_preprocess = clip.load(
            clip_name, # "ViT-B/16",
            download_root="/src/foundation_models/clip_checkpoints")
        self.clip_patch_size = self.clip_model.visual.patch_size
        self.clip_model.eval().to(config["clip_device"])

        ## Optinoal backbone compilation
        if config.get("enable_model_compilation", False):
            self.dinov2_model = torch.compile(self.dinov2_model)
            self.clip_model = torch.compile(self.clip_model)

    def encode_image(self, img: Image.Image) -> None: pass # Placeholder for unified encoder with the ability to prevent redundant work

    @torch.no_grad()
    def dinov2_pipe(self, img: Image.Image) -> Dict[str, Any]:
        with self._dino_maybe_autocast():
            img_tensor = self.t_dino(img).unsqueeze(0).to(config["dinov2_device"])
            with torch.no_grad(): x = self.dinov2_model.forward_features(img_tensor)
            return x
    
    @torch.no_grad()
    def clip_pipe(self, img: Image.Image) -> Dict[str, Any]:
        with self._clip_maybe_autocast():
            img_tensor = self.t_clip(img).unsqueeze(0).to(config["clip_device"])

            b, _, input_size_h, input_size_w = img_tensor.shape
            patch_h = input_size_h // self.clip_patch_size
            patch_w = input_size_w // self.clip_patch_size
            features = self.clip_model.get_patch_encodings(img_tensor).to(torch.float32)
            img_features = features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

            return {
                "features": img_features }

    @torch.no_grad()
    def clip_pipe_global_token(self, img: Image.Image) -> Dict[str, Any]:
        with self._clip_maybe_autocast():
            img_tensor = self.t_clip(img).unsqueeze(0).to(config["clip_device"])
            global_feature = self.clip_model.encode_image(img_tensor).squeeze().to(torch.float32)
            assert len(global_feature.shape) == 1
            return {
                "features": global_feature }
    
    @torch.no_grad()
    def clip_encode_text(self, text_query: str) -> Dict[str, Any]:
        with self._clip_maybe_autocast():
            text = clip.tokenize(text_query).to(config["clip_device"])
            text_feats = self.clip_model.encode_text(text).squeeze().to(torch.float32)
            assert len(text_feats.shape) == 1
            return {
                "features": text_feats }

    @torch.no_grad()
    def clip_global_feat_switch(self, prompt: Union[str, Image.Image]) -> Dict[str, Any]:
        """Automatically switches between text and global image features for CLIP queries."""
        if isinstance(prompt, Image.Image):
            #self.encode_image(img_prompt) # TODO (future): required here with unified encoder (a lot of computation)? 
            return self.clip_pipe_global_token(prompt)
        else: return self.clip_encode_text(prompt)

featurizer = OTASFeaturizer()

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

if config["enable_spatial"]:
    import open3d as o3d
    from scipy.spatial.transform import Rotation as R
    debug("Camera calibration pipeline instantiated")
