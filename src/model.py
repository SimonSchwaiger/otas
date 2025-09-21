from vision_utils import *

## For eval against gpu versions
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, HDBSCAN

class TorchKMeans(nn.Module):
    def __init__(self, n_clusters: int, max_iter: int = 10, enable_amp_autocast: bool = False):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.enable_amp_autocast = enable_amp_autocast

        self.register_buffer("centroids_", None)
        self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._maybe_autocast = (
            lambda: torch.amp.autocast(device_type=str(self._target_device))
            if self.enable_amp_autocast else contextlib.nullcontext()
        )

    def _ensure_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        return X.to(self._target_device)

    @torch.no_grad()
    def fit(self, X: Union[np.ndarray, torch.Tensor]):
        X = self._ensure_tensor(X)
        N, D = X.shape

        torch.manual_seed(42)
        centroids = X[torch.randperm(N)[:self.n_clusters]]

        with self._maybe_autocast():
            for _ in range(self.max_iter):
                dists = torch.cdist(X, centroids, p=2)
                labels = dists.argmin(dim=1)

                new_centroids = torch.zeros_like(centroids)
                for k in range(self.n_clusters):
                    members = X[labels == k]
                    if len(members) > 0:
                        new_centroids[k] = members.mean(dim=0)
                    else:
                        # Reinitialises dead cluster to random point
                        new_centroids[k] = X[torch.randint(0, N, (1,))]

                centroids = new_centroids

        self.centroids_ = centroids

    @torch.no_grad()
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.centroids_ is None:
            raise ValueError("TorchKMeans must be fitted before calling predict.")
        X = self._ensure_tensor(X)
        with self._maybe_autocast():
            dists = torch.cdist(X, self.centroids_, p=2)
            return dists.argmin(dim=1)

    @torch.no_grad()
    def fit_predict(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        self.fit(X)
        return self.predict(X)


class TorchPCA(nn.Module):
    def __init__(self, n_components: int, enable_amp_autocast: bool = False):
        super().__init__()
        self.n_components = n_components
        self.enable_amp_autocast = enable_amp_autocast
        self.register_buffer('mean_', None)
        self.register_buffer('components_', None)
        self._target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._maybe_autocast = (
            lambda: torch.amp.autocast(device_type=str(self._target_device))
            if self.enable_amp_autocast else contextlib.nullcontext() )
        
    def _ensure_tensor(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        return X.to(self._target_device)

    @torch.no_grad()
    def fit(self, X: Union[torch.Tensor, np.ndarray]):
        X = self._ensure_tensor(X)
        with self._maybe_autocast():
            self.mean_ = X.mean(dim=0)
            X_centered = X - self.mean_
            _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            self.components_ = Vh[:self.n_components]

    @torch.no_grad()
    def transform(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("TorchPCA instance is not fitted yet. Call 'fit' first.")
        X = self._ensure_tensor(X)
        with self._maybe_autocast():
            return (X - self.mean_) @ self.components_.T

    @torch.no_grad()
    def fit_transform(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

class language_map(nn.Module):
    def __init__(self, config: dict=config):
        super().__init__()
        self.config = config
        n_clusters = config["n_clusters"]
        n_components = config["n_components"]
        
        dim_reduction_switch = {
            "pca": PCA(n_components=n_components),
            "pca_gpu": TorchPCA(n_components=n_components, enable_amp_autocast=config["enable_amp_autocast"]),
            "kpca": KernelPCA(n_components=n_components, kernel='rbf'),
            "ica": FastICA(n_components=n_components, random_state=0, whiten='unit-variance') }
        
        cluster_model_switch = {
            "kmeans": KMeans(n_clusters=n_clusters, random_state=0),
            "kmeans_gpu": TorchKMeans(n_clusters=n_clusters, enable_amp_autocast=config["enable_amp_autocast"]),
            "gmm": GaussianMixture(n_components=n_clusters, covariance_type='full'),
            "hdbscan": HDBSCAN(min_cluster_size=18, min_samples=7, cluster_selection_epsilon=0.2) }

        self._dim_reduction_trained = False # TODO: For generalised pretraining
        self._cluster_model_trained = False
        self._dim_reduction_factory = lambda: dim_reduction_switch[config["dim_reduction_type"]]
        self._cluster_model_factory = lambda: cluster_model_switch[config["cluster_model_type"]]

        # Keep internal shape consistent
        dino_scale_factor = self.config["dino_scale_factor"]; grid_size = 16
        self.scaled_dino_dim = grid_size*dino_scale_factor

    def _ensure_tensor(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(X, np.ndarray): X = torch.from_numpy(X)
        return X.to(self.config["dinov2_device"])

    def _ensure_numpy(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
        return X

    def _dim_reduction(self, feature_list: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if not self._dim_reduction_trained:
            dim_reduction = self._dim_reduction_factory()
            if not "gpu" in self.config["dim_reduction_type"]: feature_list = self._ensure_numpy(feature_list)
            reduced_features = dim_reduction.fit_transform(feature_list)
            reduced_features = F.normalize(self._ensure_tensor(reduced_features), dim=-1)
        else: assert False, "Dim reduction pretraining not implemented yet"

        return reduced_features
    
    def _cluster_model(self, reduced_features: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if not self._cluster_model_trained:
            cluster_model = self._cluster_model_factory()
            if not "gpu" in self.config["cluster_model_type"]: reduced_features = self._ensure_numpy(reduced_features)
            cluster_labels = cluster_model.fit_predict(reduced_features)  # Shape: (num_patches,), can be torch tensor on gpu or numpy array
            cluster_labels = self._ensure_tensor(cluster_labels)
            unique_labels = torch.unique(cluster_labels, sorted=True)
            if -1 in unique_labels: cluster_labels[cluster_labels == -1] = len(unique_labels)  # Assign a new cluster label for noise
        else: assert False, "Cluster model pretraining not implemented yet"

        return cluster_labels

    def _segmentation_map(self, cluster_labels: torch.Tensor) -> torch.Tensor:
        shared_feat_resolution = self.config["shared_feat_resolution"]
        segmentation_map_tensor = cluster_labels.reshape(
            self.scaled_dino_dim, self.scaled_dino_dim).unsqueeze(0).unsqueeze(0).float() # Shape: (1, 1, H_p, W_p)
        segmentation_map_upsampled = F.interpolate(segmentation_map_tensor, size=(shared_feat_resolution, shared_feat_resolution), 
                                                   mode="nearest").squeeze(0).squeeze(0) # Upsample to target resolution
        return segmentation_map_upsampled

    @torch.no_grad()
    def dino_upscaled_feat(self, img: Image.Image) -> torch.Tensor:
        dino_scale_factor = self.config["dino_scale_factor"]
        patch_tokens = dinov2_pipe(img)["x_norm_patchtokens"]
        patch_tokens = F.normalize(patch_tokens.squeeze(0), dim=-1)
        num_patches = patch_tokens.shape[0]
        grid_size = int(math.sqrt(num_patches)) # assumes square grid
        self.scaled_dino_dim = grid_size*dino_scale_factor # Double check this is right
        ## UPSCALED DINO FEATURE MAP
        feature_map = patch_tokens.reshape(grid_size, grid_size, -1).permute(2, 0, 1)  # shape: (feat_dim, grid_size, grid_size), (C, H, W)
        feature_map_upscaled = F.interpolate(feature_map.unsqueeze(0), scale_factor=dino_scale_factor, 
                                             mode="bilinear", align_corners=False).squeeze(0) #(feat_dim, 2*grid_size, 2*grid_size)
        upscaled_grid_size = feature_map_upscaled.shape[1]  # should be dino_scale_factor * grid_size
        upscaled_features = feature_map_upscaled.permute(1, 2, 0).reshape(-1, feature_map_upscaled.shape[0]) # (HWC, then (2*grid_size)^2, feat_dim )
        return upscaled_features

    @torch.no_grad()
    def clip_shared_feat_map(self, img: Image.Image) -> torch.Tensor:
        shared_feat_resolution = self.config["shared_feat_resolution"]
        clip_feature_map = clip_pipe(img)["features"] # [1, 512, 14, 24]
        # Scale to squared grid and shared_feat_resolution
        source = clip_feature_map.squeeze(0)  # shape: [512, H_src, W_src]
        H_src, W_src = source.shape[1], source.shape[2]
        target_H, target_W = shared_feat_resolution, shared_feat_resolution  # shared_feat_resolution, shared_feat_resolution
        grid_y, grid_x = torch.meshgrid(torch.arange(target_H).to(config["clip_device"]), 
                                        torch.arange(target_W).to(config["clip_device"]), indexing='ij')
        scale_h = H_src / target_H
        scale_w = W_src / target_W
        # Nearest neighbour map target pixel to source indices
        source_y = (grid_y.float() * scale_h).floor().clamp(max=H_src - 1).long()  # shape: [shared_feat_resolution, shared_feat_resolution]
        source_x = (grid_x.float() * scale_w).floor().clamp(max=W_src - 1).long()
        # Use advanced indexing to lookup the corresponding features
        clip_lookup_feature_map = source[:, source_y, source_x].permute(1, 2, 0) # Result is [512, shared_feat_resolution, shared_feat_resolution]; then permute to [shared, shared, 512]
        return clip_lookup_feature_map

    def dino_shared_feat_map(self, img: Image.Image) -> torch.Tensor: ## Not used anymore in main pipeline -> Kept for eval code compatibility
        dino_upscaled_flat = self.dino_upscaled_feat(img)
        dino_reduced = self._dim_reduction(dino_upscaled_flat)
        cluster_labels = self._cluster_model(dino_reduced)
        return self._segmentation_map(cluster_labels)

    def masked_average_pooling_cpu(self, segmentation_map_upsampled: torch.Tensor, clip_lookup_feature_map: torch.Tensor) -> torch.Tensor:
        ## Rquires calls to cpu due to the loop. But more reliable with datatype casting
        segmentation_map_upsampled = self._ensure_tensor(segmentation_map_upsampled)
        clip_lookup_feature_map = self._ensure_tensor(clip_lookup_feature_map)
        masked_clip_feature_map = clip_lookup_feature_map.clone()  # same shape: [shared_feat_resolution, shared_feat_resolution, 512]
        unique_labels = torch.unique(segmentation_map_upsampled)
        
        for label in unique_labels:
            # Creates a mask for the current region; shape: [shared_feat_resolution, shared_feat_resolution, 1]
            mask = (segmentation_map_upsampled == label).unsqueeze(-1).float()
            masked_sum = (clip_lookup_feature_map * mask).sum(dim=(0, 1))
            count = mask.sum() # Number of pixels in the region
            if count > 0:
                avg_feature = masked_sum / count
                avg_feature /= avg_feature.norm(dim=-1, keepdim=True)
                masked_clip_feature_map[segmentation_map_upsampled == label] = avg_feature

        return masked_clip_feature_map

    def masked_average_pooling(self, segmentation_map_upsampled: torch.Tensor, clip_lookup_feature_map: torch.Tensor) -> torch.Tensor:
        ## Faster than cpu version. TODO: Debug casting issues with spatial on GPU
        segmentation_map_upsampled = self._ensure_tensor(segmentation_map_upsampled)
        clip_lookup_feature_map = self._ensure_tensor(clip_lookup_feature_map)
        H, W, C = clip_lookup_feature_map.shape
        num_pixels = H * W
    
        flat_feats = clip_lookup_feature_map.view(-1, C)  # (N, C)
        flat_labels = segmentation_map_upsampled.view(-1).long()  # (N,)
        unique_labels = torch.unique(flat_labels)
        max_label = unique_labels.max().item()
    
        # Sum and count per label
        feat_sums = torch.zeros((max_label + 1, C), device=flat_feats.device)
        counts = torch.zeros((max_label + 1, 1), device=flat_feats.device)
        feat_sums = feat_sums.index_add(0, flat_labels, flat_feats)
        counts = counts.index_add(0, flat_labels, torch.ones_like(flat_labels, dtype=torch.float).unsqueeze(1))

        # Assigns average features back using indexing
        counts = counts.clamp(min=1); avg_feats = feat_sums / counts # No division by zero
        avg_feats = avg_feats / avg_feats.norm(dim=1, keepdim=True)
        output = avg_feats[flat_labels]  # (N, C)
        return output.view(H, W, C)
    
    @torch.no_grad()
    def embed_image(self, img: Image.Image, ret_dict: bool = False) -> Union[torch.Tensor, dict]:
        dino_upscaled_flat = self.dino_upscaled_feat(img)
        dino_reduced = self._dim_reduction(dino_upscaled_flat)
        cluster_labels = self._cluster_model(dino_reduced)
        dino_shared_feat = self._segmentation_map(cluster_labels)

        clip_shared_feat = self.clip_shared_feat_map(img)
        pooled_embeddings = self.masked_average_pooling(dino_shared_feat, clip_shared_feat)
        
        if ret_dict: 
            orig_h, orig_w = img.height, img.width
            dino_feat_map = dino_upscaled_flat.reshape(self.scaled_dino_dim, self.scaled_dino_dim, -1)
            return {
                "orig_h": orig_h, "orig_w": orig_w,
                "dino_feat_map": dino_feat_map,
                "dino_clusters": dino_shared_feat, 
                "clip_shared_feat": clip_shared_feat,
                "pooled_embeddings": pooled_embeddings 
            }
        else: return pooled_embeddings

class semantic_mask(nn.Module):
    def __init__(self, config: dict=config):
        super().__init__()
        self.config = config
    
    @torch.no_grad()
    def similarity(self, shared_feature_map: torch.Tensor, pos_prompts: List[str], neg_prompts: List[str] = [""]) -> torch.Tensor:
        img_feats = shared_feature_map.permute(2, 0, 1).unsqueeze(0)
    
        positive_feats = [ clip_encode_text(prompt)["features"] for prompt in pos_prompts ]
        pos_sims = [ clip_similarity(img_feats, text_feats) for text_feats in positive_feats ]
    
        if neg_prompts != [""]:
            negative_feats = [ clip_encode_text(prompt)["features"] for prompt in neg_prompts ]
            neg_sims = [ clip_similarity(img_feats, text_feats) for text_feats in negative_feats ]
        else: neg_sims = torch.zeros_like(pos_sims[0]) # This uses the same device as pos_sims
    
        lr_sims = sum(pos_sims)/len(pos_sims) - sum(neg_sims)/(len(neg_sims) + 1e-8) # Weight sum here to deal with len(pos) != len(neg)

        sim_min, sim_max = lr_sims.min(), lr_sims.max()
        sim_range = torch.clamp(sim_max - sim_min, min=0.05) # Prevent normalisation from blowing up noise due to low similarity
        lr_sims_norm = (lr_sims - sim_min) / (sim_range + 1e-8)
        #lr_sims_norm = (lr_sims - lr_sims.min()) / (lr_sims.max() - lr_sims.min() + 1e-8) # Non-clamped version
        
        return lr_sims_norm
    
    @torch.no_grad()
    def binary_mask_interpolated(self, shared_feature_map: torch.Tensor, pos_prompts: List[str],
                    neg_prompts: List[str] = [""], threshold_value: float = .5, 
                    original_img: Optional[Image.Image] = None, orig_w: float = 0., orig_h:float = 0.,
                    ret_dict: bool = False) -> ArrayLike:

        if original_img != None: orig_h, orig_w = original_img.height, original_img.width
        assert orig_h != 0 and orig_w != 0, "Input resolution not set. Please either provide original_img or orig_w and orig_h."
        
        lr_sims_norm = self.similarity(shared_feature_map, pos_prompts, neg_prompts)
        binary_mask = (lr_sims_norm > threshold_value).float()

        binary_mask_tensor = binary_mask.unsqueeze(0).unsqueeze(0)
        binary_mask_up = F.interpolate(binary_mask_tensor, size=(orig_h, orig_w), mode='nearest')
        binary_mask_up = binary_mask_up.squeeze()
        return binary_mask_up.cpu().numpy()

    @torch.no_grad()
    def binary_mask_refined(self, shared_feature_map: torch.Tensor, pos_prompts: List[str],
                    neg_prompts: List[str] = [""], threshold_value: float = .5, 
                    original_img: Optional[Image.Image] = None, orig_w: float = 0., orig_h:float = 0.,
                    ret_dict: bool = False) -> Union[ArrayLike, dict]:

        assert self.config["enable_mask_refinement"], "Mask refinement must be enabled in config, otherwise refinement model is not loaded."
        assert original_img != None, "No original_img provided. Mask refinement requires rgb input image."
        lr_sims_norm = self.similarity(shared_feature_map, pos_prompts, neg_prompts)

        input_mask = lr_sims_norm.unsqueeze(0).unsqueeze(0)
        input_mask = F.interpolate(input_mask, size=(256, 256), mode='bilinear', align_corners=False)
        input_mask = input_mask.squeeze()  # shape: [orig_h, orig_w]
        input_mask = (input_mask*20. - 10.).cpu().numpy() # to logits
        
        upper_cutoff = 9.9999; lower_cutoff = -9.9999
        infs = input_mask[input_mask > upper_cutoff]
        input_mask[input_mask > upper_cutoff] = np.full_like(infs, upper_cutoff)
        n_infs = input_mask[input_mask < lower_cutoff]
        input_mask[input_mask < lower_cutoff] = np.full_like(n_infs, lower_cutoff)

        sam2_model.set_image(np.array(original_img.convert("RGB")))
        masks, scores, logits = sam2_model.predict(mask_input=input_mask[None, :, :])
        mask = masks[np.argmax(scores, axis=-1)] # Only best mask

        if ret_dict: return {
                "lr_sim": lr_sims_norm.cpu().numpy(), # language feature map
                "mask": mask, # predicted mask
                "input_mask": input_mask, # language-grounded mask (logits)
                "pred_logits": logits[np.argmax(scores, axis=-1)], # predicted logits from input_mask
            }
        else: return mask

class otas_spatial(nn.Module):
    def __init__(self, config: dict=config):
        super().__init__()
        self.config = config
        self._target_device = config["spatial_device"]
        self.enable_amp_autocast = config["enable_amp_autocast"]
        self.language_map_instance = language_map(config)

        self._maybe_autocast = (
            lambda: torch.amp.autocast(device_type=str(self._target_device))
            if self.enable_amp_autocast else contextlib.nullcontext() )

    def _ensure_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, np.ndarray): X = torch.from_numpy(X)
        elif isinstance(X, list): X = torch.tensor(X) # should not happen but to be sure
        return X.to(self._target_device)

    def project_3D(self, depth_array: Union[np.ndarray, torch.Tensor], 
               img_array: Union[np.ndarray, torch.Tensor], 
               fu_fv_cu_cv: Union[np.ndarray, torch.Tensor], 
               image_coords: Union[np.ndarray, torch.Tensor], 
               frame_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        depth_array = self._ensure_tensor(depth_array)
        img_array = self._ensure_tensor(img_array) 
        fu_fv_cu_cv = self._ensure_tensor(fu_fv_cu_cv)
        image_coords = self._ensure_tensor(image_coords)
    
        fu, fv, cu, cv = fu_fv_cu_cv
        u_values, v_values = image_coords[:, 0], image_coords[:, 1]
        Z = depth_array[v_values.long(), u_values.long()] # Depth values at (u, v)
        
        X = (u_values - cu) * Z / fu
        Y = (v_values - cv) * Z / fv
    
        RGB = img_array[v_values.long(), u_values.long()] # RGB values at (u, v)
        IDs = torch.full_like(Z, frame_id, device=self._target_device)
        keypoints_3d = torch.stack([X, Y, Z], dim=1)  # Shape: (M, 3), M = number of valid points
        UVs = torch.stack([u_values, v_values], dim=1)
        
        return keypoints_3d, RGB, UVs, IDs

    @staticmethod
    def pose_to_transformation_matrix(pose: List[float]) -> np.ndarray:
        assert len(pose) == 7, f"Expected pose to have 7 elements"
        x, y, z, qx, qy, qz, qw = pose
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [x, y, z]
        return T
    
    @torch.no_grad()
    def batch_embed_rgbd(
        self,
        imgs: List[Image.Image],
        depths: Union[List[ArrayLike], ArrayLike],
        poses: Union[List[ArrayLike], ArrayLike],
        fu_fv_cu_cvs: Union[List[ArrayLike], ArrayLike, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        dino_embeddings = []
        clip_maps = []
        batch_keypoints = []
        batch_RGBs = []
        batch_UVs = []
        batch_frame_IDs = []

        # Patch centers for 3d projection
        depth_h, depth_w = depths[0].shape
        patch_h = depth_h / self.config["shared_feat_resolution"]
        patch_w = depth_w / self.config["shared_feat_resolution"]
        centers_v = (patch_h / 2) + patch_h * np.arange(self.config["shared_feat_resolution"])
        centers_u = (patch_w / 2) + patch_w * np.arange(self.config["shared_feat_resolution"])
        uu, vv = np.meshgrid(centers_u.astype(int), centers_v.astype(int), indexing="xy")  # Use 'xy' indexing for Cartesian convention
        image_coords = np.stack([uu.ravel(), vv.ravel()], axis=1) # Patch centers given shared feature resolution

        if fu_fv_cu_cvs is None: fu_fv_cu_cvs = [None for _ in range(len(imgs))] # Per-frame intrinsics (optional)

        for frame_id, (img, depth_array, cam_pose, fu_fv_cu_cv) in enumerate(
            tqdm(zip(imgs, depths, poses, fu_fv_cu_cvs), total=len(imgs),
                desc="[OTAS]: Semantic Embedding and 3D Projection" )):
        
            img = img.convert("RGB")
            img_array = np.array(img)

            if fu_fv_cu_cv is None: fu, fv, cu, cv = self.config["cam0_params"]["fu_fv_cu_cv"]
            else: fu, fv, cu, cv = fu_fv_cu_cv
            
            ## Embed DINO (context managed by lang map)
            dino_embedding = self.language_map_instance.dino_upscaled_feat(img)
            dino_embeddings.append(dino_embedding.cpu())
        
            ## Embed CLIP (context managed by lang map)
            clip_embedding = self.language_map_instance.clip_shared_feat_map(img)
            clip_maps.append(clip_embedding.unsqueeze(0).cpu())
            
            ## Project 3D (in shared feature space with nearest downsampling)
            with self._maybe_autocast():
                # Set depth values over threshold to infinity
                depth_array = np.array(depth_array)  # Ensure numpy array
                depth_array[depth_array > self.config.get("spatial_max_depth", 1000.0)] = np.inf
                # Sample median depth per patch
                depth_h, depth_w = depth_array.shape
                depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0)
                depth_resized = F.interpolate(depth_tensor, mode='nearest',
                                              size=(self.config["shared_feat_resolution"], self.config["shared_feat_resolution"]))
                depth_resized = F.interpolate(depth_resized, mode='nearest', size=(depth_h, depth_w)) # Lazily cast back to orig resolution to adhere to cam model
                depth_nearest = depth_resized.squeeze(0).squeeze(0)
            
                ## Transform keypoints to common world coordinate frame, equivalent to o3d.pcd.transform but faster on gpu. This will take some vram
                keypoints_3d, RGBs, UVs, IDs = self.project_3D(depth_nearest, img_array, np.array([fu, fv, cu, cv]), image_coords, frame_id = frame_id)
                pts = keypoints_3d.to(self._target_device)
                ones = torch.ones((pts.shape[0], 1), device=pts.device)
                pts_h = torch.cat([pts, ones], dim=1)
                T_c2w = torch.tensor(self.pose_to_transformation_matrix(cam_pose), device=pts.device, dtype=pts_h.dtype)
                pts_transformed_h = (T_c2w @ pts_h.T).T
                pts_transformed = pts_transformed_h[:, :3]
            
            batch_keypoints.append(pts_transformed.cpu()) # Store everything on cpu while not doing anything to save vram
            batch_RGBs.append(RGBs.cpu())
            batch_UVs.append(UVs.cpu())
            batch_frame_IDs.append(IDs.cpu())
            
        dino_embeddings = torch.cat(dino_embeddings, dim=0) # (num_imgs*shared_dim^2, 384)
        clip_maps = torch.cat(clip_maps, dim=0) # (num_imgs, shared_dim, shared_dim, 512)
        batch_keypoints = torch.cat(batch_keypoints, dim=0)
        batch_RGBs = torch.cat(batch_RGBs, dim=0)
        batch_UVs = torch.cat(batch_UVs, dim=0)
        batch_frame_IDs = torch.cat(batch_frame_IDs, dim=0).to(int) # (num_imgs*shared_dim^2,)

        return dino_embeddings, clip_maps, batch_keypoints, batch_RGBs, batch_UVs, batch_frame_IDs

    def batch_pool_embeddings(
        self,
        dino_embeddings: torch.Tensor,
        clip_maps: torch.Tensor,
        batch_keypoints: torch.Tensor,
        batch_RGBs: torch.Tensor,
        batch_UVs: torch.Tensor,
        batch_frame_IDs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        valid_entries = torch.logical_not(torch.any(torch.isnan(batch_keypoints), 1))
        valid_dino_embeddings = dino_embeddings[valid_entries]
        valid_keypoints = batch_keypoints[valid_entries]
        norm_keypoints = F.normalize(valid_keypoints, dim=-1)

        ## REDUCE
        combined_point_feats = torch.cat([valid_dino_embeddings, norm_keypoints], dim=1)
        combined_point_feats = F.normalize(combined_point_feats, dim=-1)
        reduced_point_feats = self.language_map_instance._dim_reduction(combined_point_feats)
        
        ## CLUSTER
        norm_keypoints = self._ensure_tensor(norm_keypoints)
        valid_entries = self._ensure_tensor(valid_entries)
        combined_reduced_feats = torch.cat([reduced_point_feats, norm_keypoints], dim=1)
        cluster_labels = self.language_map_instance._cluster_model(reduced_point_feats)

        ## BACKPROJECT AND MASKED POOLING
        pooled_embeddings = []
        global_labels = torch.tensor(np.full(len(batch_keypoints), -1), 
                                     dtype=cluster_labels.dtype, device=self._target_device) # Line up indices with preprocessed patches
        global_labels[valid_entries] = cluster_labels
        
        unique_frame_ids = np.unique(batch_frame_IDs.cpu().numpy())
        for frame_id in tqdm(
            unique_frame_ids, total=len(unique_frame_ids),
            desc="[OTAS]: Backprojecting and pooling embeddings per frame" ):
            
            start = frame_id*self.config["shared_feat_resolution"]**2
            end = (frame_id + 1)*self.config["shared_feat_resolution"]**2
            
            labels = global_labels[start:end]
            shared_dino_feat = self.language_map_instance._segmentation_map(labels)
            shared_clip_feat = clip_maps[frame_id]
            pooled_embedding = self.language_map_instance.masked_average_pooling_cpu(shared_dino_feat, shared_clip_feat)
            pooled_embedding_flat = torch.reshape(pooled_embedding, (self.config["shared_feat_resolution"]**2, -1))
            pooled_embeddings.append(pooled_embedding_flat.cpu())
        
        pooled_embeddings = torch.cat(pooled_embeddings, dim=0)

        return pooled_embeddings, global_labels.cpu(), valid_entries.cpu()

    @torch.no_grad()
    def batch_voxelise(self, pooled_embeddings: torch.Tensor, batch_keypoints: torch.Tensor, valid_entries: torch.Tensor) -> Union[Any, torch.Tensor, ArrayLike]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(batch_keypoints[valid_entries].cpu().numpy())
        pcd_downsampled, _, trace = pcd.voxel_down_sample_and_trace(
            self.config["voxel_size"], pcd.get_min_bound(), pcd.get_max_bound())
        
        # Bring trace into usable form (default lookup is really slow
        max_len = max(len(lst) for lst in trace)
        trace_arr = np.full((len(trace), max_len), -1, dtype=int)
        for line_idx, line in enumerate(trace):
            trace_arr[line_idx][0:len(line)] = np.array(line)
    
        num_points_down = np.array(pcd_downsampled.points).shape[0]
        vl_dim = pooled_embeddings.shape[-1]
        pooled_embeddings_geo = torch.empty((num_points_down, vl_dim), device="cpu")
        valid_indices = np.where(valid_entries)[0]
    
        for trace_idx, entry in tqdm(enumerate(trace_arr), total=len(trace_arr), desc="[OTAS]: Performing Voxelisation"):
            valid_pcd_idx = entry[entry != -1] # valid pcd idx
            global_idx = valid_indices[valid_pcd_idx] # global pcd idx
            voxel_feats = pooled_embeddings[global_idx] # tensor of related feats
            # Pool within voxel
            pooled_features = voxel_feats.mean(dim=0, keepdim=True)  # Shape: [1, 512]
            pooled_features = F.normalize(pooled_features, p=2, dim=1)  # L2 along feat. to retain language-grounding
            pooled_embeddings_geo[trace_idx] = pooled_features

        return pcd_downsampled, pooled_embeddings_geo, trace_arr

    @torch.no_grad()
    def similarity(self, pooled_embeddings: torch.Tensor, pos_prompts: List[str], neg_prompts: List[str] = [""]) -> torch.Tensor:
        pooled_embeddings = self._ensure_tensor(pooled_embeddings)
        positive_feats = [ clip_encode_text(prompt)["features"] for prompt in pos_prompts ]
        pos_sims = [ clip_similarity_flat(pooled_embeddings, text_feats) for text_feats in positive_feats ]
        
        if neg_prompts != [""]:
            negative_feats = [ clip_encode_text(prompt)["features"] for prompt in neg_prompts ]
            neg_sims = [ clip_similarity_flat(pooled_embeddings, text_feats) for text_feats in negative_feats ]
        else: neg_sims = torch.zeros_like(pos_sims[0]) # This uses the same device as pos_sims
        
        lr_sims = sum(pos_sims)/len(pos_sims) - sum(neg_sims)/len(neg_sims)
        lr_sims_norm = (lr_sims - lr_sims.min()) / (lr_sims.max() - lr_sims.min() + 1e-8)
        return lr_sims_norm.cpu()