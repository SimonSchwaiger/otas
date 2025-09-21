## Basic logging
TRANSFORM_SHAPE = (7,) # Standard transform format (x, y, z, x, y, z, w)
DEBUG_MSGS = True
USE_ROS_DEBUG = False
if USE_ROS_DEBUG: import rospy

def debug(msg, print_debug: bool = DEBUG_MSGS, use_ros: bool = USE_ROS_DEBUG) -> None:
    """Prints debug messages to stdout or to ros logging"""
    if print_debug != True: return
    if use_ros == True: rospy.loginfo(msg) #TODO: ROS 2
    else: print(f"[OTAS]: {msg}")

## DL hardware
TORCH_DEVICES = ["cpu", "cuda"]

## Check of configurable model calls
from inspect import Signature
from typing import Callable, Any, Dict, Union, List

def check_and_call(func: Callable[..., Any], params: Dict[str, Any], function_signature: Signature) -> Any:
    """Checks if a provided config is valid for a provided callable. If provided config is valid, the function is called, otherwise an exception is thrown."""
    valid_keys = function_signature.parameters.keys()

    # Raise exception if invalid configuration is provided
    invalid_keys = [key for key in params if key not in valid_keys]
    if invalid_keys: raise ValueError(f"Invalid parameter(s) provided: {', '.join(invalid_keys)}")

    # Call the function with valid parameters and return
    valid_params = {key: value for key, value in params.items() if key in valid_keys}
    return func(**valid_params)

## Check camera input params
from numpy.typing import ArrayLike
import numpy as np

DISTORTION_MODELS = ["radial-tangential"]

def check_camera(fu_fv_cu_cv: ArrayLike, dist_coeffs: ArrayLike, distortion_model: str) -> None:
    assert len(fu_fv_cu_cv) == 4
    assert len(dist_coeffs) == 4
    assert distortion_model in DISTORTION_MODELS

## Check 3D projection parameters. Specifically, if provided coordinates are within provided depth image
def check_3d_projection(depth_array: ArrayLike, fu_fv_cu_cv: Union[List[float], ArrayLike], image_coords: ArrayLike) -> None:
    if not isinstance(fu_fv_cu_cv, np.ndarray): fu_fv_cu_cv = np.array(fu_fv_cu_cv)
    assert depth_array.ndim == 2; assert fu_fv_cu_cv.shape == (4,) # depth (u, v), fu_fv_cu_cv (4,)
    assert image_coords.ndim == 2 and image_coords.shape[1] == 2 # image_coords (n, 2)
    assert np.all(depth_array.shape[::-1] > np.max(image_coords, axis=0)), f"image_coords exceed provided depth image!"