import os; import sys
import json
from pathlib import Path
import copy
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F

import open3d as o3d
from PIL import Image

from typing import List, Union, Tuple
from numpy.typing import ArrayLike

#if os.isatty(sys.stdout.fileno()): current_dir = os.getcwd() # Paths will be relative to PWD if running in a terminal
current_dir = os.path.dirname(os.path.abspath(__file__))

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc # Force garbage collection
        gc.collect()

##============================
## Single View (2D)

#def batchify(X, batch_size):
#    for i in range(0, len(X), batch_size):
#        yield X[i:i + batch_size]

class single_inference:
    def __init__(self, config_path=f"{current_dir}/model_config/OTAS_small.json", **kwargs):
        os.environ["OTAS_CONFIG_PATH"] = str(Path(config_path).expanduser().resolve())
        import model
        
        self.base_config = copy.deepcopy(model.config)
        self.config = self.base_config; self.config.update(kwargs)
        self.language_map_instance = model.language_map(config=self.config)
        self.mask_instance = model.semantic_mask(config=self.config)

    def similarity_single(self, img: Image.Image, pos_prompts: List[str], neg_prompts: List[str] = [""]):
        vlm_embedding = self.language_map_instance.embed_image(img)
        lr_sims = self.mask_instance.similarity(vlm_embedding, pos_prompts, neg_prompts=neg_prompts)
        orig_h, orig_w = img.height, img.width
        lr_sims_tensor = lr_sims.unsqueeze(0).unsqueeze(0)
        lr_sims_up = F.interpolate(lr_sims_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return lr_sims_up.squeeze().cpu().numpy()

    def segmentation_single(self, img: Image.Image, pos_prompts: List[str], neg_prompts: List[str] = [""], threshold_value: float = 0.5):    
        vlm_embedding = self.language_map_instance.embed_image(img)
        if self.config["enable_mask_refinement"]:
            mask = self.mask_instance.binary_mask_refined(vlm_embedding, pos_prompts, neg_prompts=neg_prompts,
                                                    original_img=img, ret_dict=False, threshold_value=threshold_value)
        else: 
            mask = self.mask_instance.binary_mask_interpolated(vlm_embedding, pos_prompts, neg_prompts=neg_prompts,
                                                    original_img=img, ret_dict=False, threshold_value=threshold_value)
        return mask
    
    def similarity_batch(self, imgs: List[Image.Image], pos_prompts: List[str],
                         neg_prompts: List[str] = [""]):
        predictions = []
        for img in tqdm(imgs, desc="[OTAS]: Similarity Batch"):
            predictions.append(self.similarity_single(img, pos_prompts, neg_prompts))
        
        return predictions

    def segmentation_batch(self, imgs: List[Image.Image], pos_prompts: List[str],
                           neg_prompts: List[str] = [""], threshold_value: float = 0.5):
        predictions = []
        for img in tqdm(imgs, desc="[OTAS]: Segmentation Batch"):
            predictions.append(self.segmentation_single(img, pos_prompts, neg_prompts, threshold_value))
        
        return predictions

##============================
## Multi View (3D)

def all_imgs_in_dir(directory: str, extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif')) -> List[str]:
    directory_path = Path(directory)
    return sorted([
        str(p) for p in directory_path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions ])

def equally_space(paths: List[str], n_images: int) -> List[str]:
    if len(paths) <= n_images: return paths
    indices = np.linspace(0, len(paths)-1, n_images, dtype=int)
    return [paths[i] for i in indices]

def tf_mat2pose(tf_mat: ArrayLike) -> ArrayLike:
    """Convert a 4x4 transformation matrix to a 7-parameter pose vector"""
    # Extract translation
    x, y, z = tf_mat[:3, 3]
    qx, qy, qz, qw = R.from_matrix(tf_mat[:3, :3]).as_quat()
    return np.array([x, y, z, qx, qy, qz, qw])

def vggt_inference(img_paths: List[str]):
    """Helper to estimate extrinsics and depth if not known"""
    import sys; sys.path.append(f"{current_dir}/foundation_models/vggt")
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    print(f"[OTAS]: No depth or camera poses provided, estimating from VGGT")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu" #TODO: Make configurable, because vggt takes a lot of VRAM
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    if device == "cuda": dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else: dtype = torch.float16
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    images = load_and_preprocess_images(img_paths).to(device)

    try:
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=dtype):
                predictions = vggt_model(images)

            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:]) # see https://github.com/facebookresearch/vggt/issues/18
            predictions["extrinsic"] = extrinsic # 3x3 Rot mat and 1x3 trans vec for each frame
            predictions["intrinsic"] = intrinsic # 3x3 Intrinsic mat for each frame
            
            fu_fv_cu_cv = []
            for i in range(intrinsic.shape[1]):
                intr = intrinsic[0,i].cpu().numpy()
                fu_fv_cu_cv.append([intr[0,0], intr[1,1], intr[0,2], intr[1,2]])

            # VGGT provides w2c extrinsics, so we need to convert to c2w
            cam_poses = []
            for i in range(extrinsic.shape[1]):
                w2c = np.vstack((extrinsic[0][i].cpu().numpy(), np.array([0,0,0,1])))
                c2w = np.linalg.inv(w2c)
                cam_poses.append(tf_mat2pose(c2w))

            depths = [depth.squeeze().cpu().numpy() for depth in predictions["depth"].squeeze(0).squeeze(-1)]
            ret_images = [Image.fromarray((img.transpose(1,2,0)*255).astype(np.uint8)) for img in predictions["images"].squeeze(0).cpu().numpy()] # to PIL
    finally:
        # Explicitly delete all GPU tensors
        del vggt_model; del images; del predictions; del extrinsic; del intrinsic
        clear_cuda_cache()

    ## Segment sky to remove depth outliers. see https://github.com/facebookresearch/vggt/blob/main/vggt_to_colmap.py
    print(f"[OTAS]: Masking sky for depth estimation")
    enable_autocast = False if device == "cpu" else True
    otas_instance = single_inference(enable_amp_autocast=enable_autocast)
    sky_masks = otas_instance.segmentation_batch(ret_images, ["sky", "clouds"], neg_prompts=["ground", "object"], threshold_value=0.24)

    for depth, sky_mask in zip(depths, sky_masks): # mask sky as invalid #TODO: Maybe inflate mask just to be safe
        depth[sky_mask >= 0.5] = np.inf

    ## Scale depths to metric depth
    print(f"[OTAS]: Scaling depths to metric depth")
    vggt_depth_first = depths[0]
    zoedepth_depth = zoedepth_inference(img_paths[0], resize_shape=(vggt_depth_first.shape[1], vggt_depth_first.shape[0]))
    
    # Compute ratio for each pixel where both depth maps have valid values
    valid_mask = (zoedepth_depth > 0) & (vggt_depth_first > 0) & (vggt_depth_first < np.inf)
    if valid_mask.any():
        ratios = zoedepth_depth[valid_mask] / vggt_depth_first[valid_mask]
        scale_factor = np.median(ratios)
    else:
        scale_factor = 1.0 # Fallback if no valid pixels, this should not happen

    scale_factor = scale_factor / 10. #TODO: debug scale shift
    
    metric_depths = []
    for depth in depths:
        metric_depth = depth * scale_factor
        metric_depths.append(metric_depth)
    
    # Scale the translation vectors in the camera poses
    scaled_cam_poses = []
    for pose in cam_poses:
        translation = pose[:3]
        rotation = pose[3:]
        
        scaled_translation = translation * scale_factor
        
        scaled_pose = np.concatenate([scaled_translation, rotation])
        scaled_cam_poses.append(scaled_pose)

    return ret_images, metric_depths, scaled_cam_poses, fu_fv_cu_cv

def zoedepth_inference(img_path: str, resize_shape: Union[Tuple[int, int], None] = None):
    from transformers import pipeline
    try:
        pipe = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")
        img = Image.open(img_path)
        zoedepth_img = pipe(img)["depth"]
        if resize_shape is not None:
            zoedepth_img = zoedepth_img.resize(resize_shape, Image.Resampling.NEAREST)
        zoedepth_array = np.array(zoedepth_img)
    finally:
        del pipe
        clear_cuda_cache()

    return zoedepth_array
    
class spatial_inference:
    """Performs OTAS 3D Reconstruction from a nerfstudio dataset or plain images"""
    def __init__(self, config_path=f"{current_dir}/model_config/OTAS_spatial.json",
                 ns_data_path: Union[str, None] = None,
                 vggt_image_paths: Union[List[str], None] = None,
                 **kwargs):
        os.environ["OTAS_CONFIG_PATH"] = str(Path(config_path).expanduser().resolve())
        import model
        
        self.base_config = copy.deepcopy(model.config)
        self.config = self.base_config; self.config.update(kwargs)

        self._model_factory = lambda **kwargs: model.otas_spatial(**kwargs)

        if ns_data_path is not None: self.initialise_from_ns_data(ns_data_path)
        elif vggt_image_paths is not None: self.initialise_from_vggt_data(vggt_image_paths)
        else: self.spatial_instance = self._model_factory(config=self.config)
        
    def initialise_from_vggt_data(self, vggt_image_paths: List[str]) -> None:    
        img_paths = equally_space(vggt_image_paths, self.config.get("spatial_batch_size", len(vggt_image_paths)))
        
        imgs, depths, cam_poses, fu_fv_cu_cvs = vggt_inference(img_paths)
        self.imgs = imgs; self.depths = depths; self.cam_poses = cam_poses; self.fu_fv_cu_cvs = fu_fv_cu_cvs

        self.spatial_instance = self._model_factory(config=self.config)
        self.reconstruct(imgs, depths, cam_poses, fu_fv_cu_cvs=fu_fv_cu_cvs)
        
    def initialise_from_ns_data(self, ns_data_path: str) -> None:
        ns_data_path = Path(ns_data_path).expanduser().resolve()
        assert ns_data_path.exists(), f"Nerfstudio data path does not exist: {ns_data_path}"
        assert ns_data_path.name == "transforms.json", f"Expected 'transforms.json', got '{ns_data_path.name}'"
        with open(ns_data_path, 'r') as f: ns_data = json.load(f)
        ## Extract intrinsics -> They may be defined globally or per frame we distinguish on the keys being present globally
        try:
            # Global intrinsics
            fl_x = ns_data['fl_x']; fl_y = ns_data['fl_y']
            cx = ns_data['cx']; cy = ns_data['cy']
            width = ns_data['w']; height = ns_data['h']
            per_frame_fu_fv_cu_cv = None
        except KeyError:
            ## Per frame intrinsics
            per_frame_fu_fv_cu_cv = []
            for entry in ns_data['frames']:
                fl_x = entry['fl_x']; fl_y = entry['fl_y']
                cx = entry['cx']; cy = entry['cy']
                width = entry['w']; height = entry['h']
                per_frame_fu_fv_cu_cv.append([fl_x, fl_y, cx, cy])
            
        if per_frame_fu_fv_cu_cv == None: # Only for global intrinsics
            fov = ns_data.get('camera_angle_x', None)
            cam0_params = {
                "fu_fv_cu_cv": [fl_x, fl_y, cx, cy],
                "width": width,
                "height": height, }
            if fov is not None:
                import math
                cam0_params["fov"] = round(fov * 180 / math.pi, 2)  # rad2deg
            self.config["cam0_params"] = cam0_params

        img_paths = []; depth_paths = []; cam_poses = []
        for entry in ns_data['frames']:
            ns_data_dir = str(Path(ns_data_path).parent)
            img_paths.append(str(Path(ns_data_dir) / entry['file_path']))
            flip_y = np.diag([1, -1, -1, 1])
            tf = np.array(entry['transform_matrix'])
            tf = flip_y @ tf @ flip_y
            cam_poses.append(tf_mat2pose(tf))

            if 'depth_path' in entry:
                depth_paths.append(str(Path(ns_data_dir) / entry['depth_path']))

        img_paths = equally_space(img_paths, self.config.get("spatial_batch_size", len(img_paths)))
        depth_paths = equally_space(depth_paths, self.config.get("spatial_batch_size", len(depth_paths)))
        cam_poses = equally_space(cam_poses, self.config.get("spatial_batch_size", len(cam_poses)))

        imgs = [Image.open(img_path) for img_path in img_paths]

        if len(depth_paths) != len(img_paths):
            print("[OTAS]: Missing depth maps, estimating using ZoeDepth")
            from transformers import pipeline
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            depth_estimator = pipeline("depth-estimation", model="facebook/zoedepth-nk", device=device)
            
            try:
                depths = []
                for img in imgs:
                    with torch.amp.autocast(device_type=device, dtype=dtype):
                        depth = depth_estimator(img)["predicted_depth"]
                        depths.append(depth)
            finally:
                # Clean up GPU memory
                del depth_estimator
                clear_cuda_cache()
        else:
            depths = [np.array(Image.open(depth_path)) / 1000.0 for depth_path in depth_paths] # mm to meters

        self.spatial_instance = self._model_factory(config=self.base_config)
        self.reconstruct(imgs, depths, cam_poses, per_frame_fu_fv_cu_cv)

    def export_to_ns_data(self, ns_data_out_path: str) -> None:
        """Exports the reconstructed point cloud to a nerfstudio dataset"""
        ns_data_out_path = Path(ns_data_out_path).expanduser().resolve()
        ns_data_out_path.mkdir(parents=True, exist_ok=True)
        
        images_dir = ns_data_out_path / "images"
        depth_dir = ns_data_out_path / "depth"
        images_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)
        
        ns_data = { "camera_model": "OPENCV", "frames": [] }
        
        for i, (img, depth, pose) in enumerate(zip(self.imgs, self.depths, self.cam_poses)):
            img_filename = f"frame_{i:06d}.jpg"
            img_path = f"images/{img_filename}"  # Path relative to transforms.json
            img.save(images_dir / img_filename)
            
            depth_filename = f"frame_{i:06d}.png"
            depth_path = f"depth/{depth_filename}"
            
            # Handle invalid depth values before conversion
            depth_valid = depth.copy()
            depth_valid[~np.isfinite(depth_valid)] = 0  # Set NaN and inf to 0
            depth_mm = (depth_valid * 1000).astype(np.uint16)  # m to mm
            Image.fromarray(depth_mm).save(depth_dir / depth_filename)
            
            # Convert pose to transformation matrix
            x, y, z, qx, qy, qz, qw = pose
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot
            transform_matrix[:3, 3] = [x, y, z]

            flip_y = np.diag([1, -1, -1, 1])
            transform_matrix = flip_y @ transform_matrix @ flip_y
            transform_matrix = [[float(x) for x in row] for row in transform_matrix] # all float
            
            if hasattr(self, 'fu_fv_cu_cvs') and self.fu_fv_cu_cvs is not None:
                if isinstance(self.fu_fv_cu_cvs, list):
                    fu_fv_cu_cv = self.fu_fv_cu_cvs[i]  # Use this frame's parameters
                else:
                    fu_fv_cu_cv = self.fu_fv_cu_cvs
                fl_x, fl_y, cx, cy = fu_fv_cu_cv
            else:
                # No override, use config.json values
                fl_x = self.config["cam0_params"]["fu_fv_cu_cv"][0]
                fl_y = self.config["cam0_params"]["fu_fv_cu_cv"][1]
                cx = self.config["cam0_params"]["fu_fv_cu_cv"][2]
                cy = self.config["cam0_params"]["fu_fv_cu_cv"][3]
            
            width = img.width
            height = img.height
            
            frame_data = {
                "file_path": img_path,
                "depth_path": depth_path,
                "transform_matrix": transform_matrix,
                "fl_x": float(fl_x),
                "fl_y": float(fl_y),
                "cx": float(cx),
                "cy": float(cy),
                "w": int(width),
                "h": int(height)
            }
            ns_data["frames"].append(frame_data)
        
        with open(ns_data_out_path / "transforms.json", 'w') as f:
            json.dump(ns_data, f, indent=2)

    def reconstruct(self, imgs: List[Image.Image], depths: List[ArrayLike], cam_poses: List[ArrayLike], fu_fv_cu_cvs: Union[List[ArrayLike], ArrayLike, None] = None) -> None:
        
        if self.config.get("log_time", False):
            import time
            start_time = time.time()

        (self.dino_embeddings, self.clip_maps, self.batch_keypoints, 
         self.batch_RGBs, self.batch_UVs, self.batch_frame_IDs) = \
            self.spatial_instance.batch_embed_rgbd(imgs, depths, cam_poses, fu_fv_cu_cvs)

        (self.pooled_embeddings, self.global_cluster_labels, 
         self.valid_entries) = self.spatial_instance.batch_pool_embeddings(
            self.dino_embeddings, self.clip_maps, self.batch_keypoints,
            self.batch_RGBs, self.batch_UVs, self.batch_frame_IDs)

        (self.pcd, self.pooled_embeddings_geo, 
         self.trace_arr) = self.spatial_instance.batch_voxelise(
            self.pooled_embeddings, self.batch_keypoints, self.valid_entries)

        if self.config.get("log_time", False): print(f"Reconstruction time: {time.time() - start_time:.2f} seconds")

        self.pcd_colour = o3d.geometry.PointCloud()
        points = self.batch_keypoints.cpu().numpy()[self.valid_entries.cpu().numpy()]
        self.pcd_colour.points = o3d.utility.Vector3dVector(points)
        colour_rgb = self.batch_RGBs.cpu().numpy()[self.valid_entries.cpu().numpy()]
        self.pcd_colour.colors =  o3d.utility.Vector3dVector(colour_rgb.astype(float)/255)

    def visualise_pca_pcd(self) -> o3d.geometry.PointCloud:
        """Returns pcd with PCA-reduced pooled embeddings for visualisation"""
        assert hasattr(self, 'pcd') and hasattr(self, 'pooled_embeddings_geo'), "Must call reconstruct() before querying the point cloud"
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3) # For rgb
        valid_embeds = self.pooled_embeddings.cpu().numpy()[self.valid_entries.cpu().numpy()]
        reduced_rgb = pca.fit_transform(valid_embeds)
        colour_pca = F.normalize(torch.tensor(reduced_rgb).unsqueeze(0).unsqueeze(0), dim=-1).squeeze().squeeze().cpu().numpy()
        pcd = o3d.geometry.PointCloud(self.pcd_colour)
        pcd.colors = o3d.utility.Vector3dVector(colour_pca)
        return pcd
    
    def visualise_cluster_pcd(self) -> o3d.geometry.PointCloud:
        """Returns pcd with coloured points based on cluster labels"""
        assert hasattr(self, 'pcd') and hasattr(self, 'global_cluster_labels'), "Must call reconstruct() before querying the point cloud"
        cluster_labels = self.global_cluster_labels.cpu().numpy()[self.valid_entries.cpu().numpy()]
        lower, upper = cluster_labels.min(), cluster_labels.max()
        norm = (cluster_labels.astype(float) - lower) / (upper - lower)
        from matplotlib import colormaps as cm
        viridis = cm.get_cmap('viridis')
        colour_cluster = np.array([ viridis(entry) for entry in norm ]) # 0 = no valid cluster/projection
        pcd = o3d.geometry.PointCloud(self.pcd_colour)
        pcd.colors = o3d.utility.Vector3dVector(colour_cluster[:,:3])
        return pcd

    def query_masked_pcd(self, pos_prompts: List[str], neg_prompts: List[str], threshold: float = 0.5) -> o3d.geometry.PointCloud:
        """Returns pcd only containing points with semantic similarity greater than threshold"""
        assert hasattr(self, 'pcd') and hasattr(self, 'pooled_embeddings_geo'), "Must call reconstruct() before querying the point cloud"
        similarity = self.spatial_instance.similarity(self.pooled_embeddings_geo, pos_prompts, neg_prompts)
        mask = similarity > threshold
        indices = np.where(mask.cpu().numpy())[0]
        pcd = self.pcd.select_by_index(indices)
        return pcd
    
    def query_similarity_pcd(self, pos_prompts: List[str], neg_prompts: List[str], cmap: str = 'Reds', threshold: float = -1000.) -> o3d.geometry.PointCloud:
        """Returns pcd with coloured points based on semantic similarity. Removes points with similarity less than threshold."""
        assert hasattr(self, 'pcd') and hasattr(self, 'pooled_embeddings_geo'), "Must call reconstruct() before querying the point cloud"
        similarity = self.spatial_instance.similarity(self.pooled_embeddings_geo, pos_prompts, neg_prompts)
        from matplotlib import colormaps as cm
        cmap = cm.get_cmap(cmap)
        colour_relevance = np.array([ cmap(entry) if entry > threshold else [0,0,0,0] for entry in similarity.cpu().numpy() ])
        pcd = o3d.geometry.PointCloud(self.pcd)
        pcd.colors = o3d.utility.Vector3dVector(colour_relevance[:,:3])
        return pcd

    @staticmethod
    def cleanup_overexposed_pcd(pcd: o3d.geometry.PointCloud, brightness_threshold: float = 0.75) -> o3d.geometry.PointCloud:
        """Returns a pcd without overexposed points. All points with higher brightness than brightness_threshold are removed"""
        colours = np.asarray(pcd.colors)
        not_overexposed = np.any(colours < brightness_threshold, axis=1)
        return pcd.select_by_index(np.where(not_overexposed)[0])

    @staticmethod
    def cleanup_geometric_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Returns a pcd with statistical and radius outliers removed"""
        cl, ind = pcd.remove_radius_outlier(nb_points=9, radius=4)
        pcd_rad = pcd.select_by_index(ind)
        cl, ind = pcd_rad.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.8)
        return pcd_rad.select_by_index(ind)

    @staticmethod
    def cleanup_semantic_pcd(semantic_pcd: o3d.geometry.PointCloud, geometric_pcd: o3d.geometry.PointCloud, distance_threshold: float = 0.5) -> o3d.geometry.PointCloud:
        """Returns semantic pcd with points only within distance_threshold [m] of the geometric pcd"""
        filtered_combined_pcd = o3d.geometry.PointCloud()
        pcd_tree = o3d.geometry.KDTreeFlann(geometric_pcd)
        points = np.asarray(semantic_pcd.points)
        filtered_points = []
        filtered_colors = []

        for i, point in enumerate(points):
            # Find closest point in rgb_pcd
            [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
            if k > 0:
                # Get distance to closest point
                closest_point = np.asarray(geometric_pcd.points)[idx[0]]
                dist = np.linalg.norm(point - closest_point)
                # Only keep points within 0.5m of rgb surface
                if dist < distance_threshold:
                    filtered_points.append(point)
                    if semantic_pcd.has_colors(): filtered_colors.append(np.asarray(semantic_pcd.colors)[i])

        filtered_combined_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
        if semantic_pcd.has_colors(): 
            filtered_combined_pcd.colors = o3d.utility.Vector3dVector(np.array(filtered_colors))
        return filtered_combined_pcd