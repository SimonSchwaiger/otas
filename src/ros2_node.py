# ROS 2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

# ROS types
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2, PointField, Image as ROSImage, CameraInfo
from geometry_msgs.msg import TransformStamped

# ROS conversions
from sensor_msgs_py import point_cloud2 as pc2
import message_filters
from cv_bridge import CvBridge
_bridge = CvBridge()

import tf2_ros
from tf2_ros import TransformException

# Python
import numpy as np
import open3d as o3d
from ctypes import * # convert float to uint32
from PIL import Image

import threading
from copy import deepcopy

from typing import List, Tuple, Optional, Dict, Any

# OTAS
from inference import spatial_inference, clear_cuda_cache, equally_space

##------------------
## Point Cloud Conversion

## FROM: https://github.com/IASRobolab/o3d_ros_point_cloud_converter/blob/master/o3d_ros_point_cloud_converter/conversion.py
# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def pcl2roscloud(open3d_cloud, frame_id="base_link"):
    # Set "header"
    header = Header()
    header.stamp = rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points = np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points.tolist()
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255).astype('uint32') # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]
        points = points.tolist()
        colors = colors.tolist()
        cloud_data = []
        for i in range(len(points)):
            cloud_data.append(points[i]+[colors[i]])

    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

##------------------
## Publishes clouds from OTAS Spatial to ROS 2

class OTASVisualizationNode(Node):
    """Basic ROS2 node for publishing OTAS semantic pointclouds"""
    
    def __init__(self, automatic_spin=True): # internal spinning set to True for convenience when just visualising from a notebook
        super().__init__('otas_visualization_node')
        self.automatic_spin = automatic_spin
        
        self.geo_publisher = self.create_publisher(PointCloud2, '/otas/pointcloud/geometric', 1)
        self.pca_publisher = self.create_publisher(PointCloud2, '/otas/pointcloud/pca', 1)
        self.sim_publisher = self.create_publisher(PointCloud2, '/otas/pointcloud/similarity', 1)
        self.mask_publisher = self.create_publisher(PointCloud2, '/otas/pointcloud/mask', 1)
        self.cluster_publisher = self.create_publisher(PointCloud2, '/otas/pointcloud/cluster', 1)

        self.pcd_geo = None
        self.pcd_pca = None
        self.pcd_sim = None
        self.pcd_mask = None
        self.pcd_cluster = None
    
    def pub_geometric(self, model, store_pcd: bool = True, do_cleanup: bool = True, frame_id: str = "base_link") -> None:
        pcd_geo = model.cleanup_overexposed_pcd(model.pcd_colour)
        if do_cleanup: pcd_geo = model.cleanup_geometric_pcd(pcd_geo)
        if store_pcd: self.pcd_geo = pcd_geo
        self.publish_pointcloud(pcd_geo, self.geo_publisher, frame_id)

    def pub_pca(self, model, store_pcd: bool = True, do_cleanup: bool = True, frame_id: str = "base_link") -> None:
        pcd_pca = model.visualise_pca_pcd()
        if do_cleanup: pcd_pca = model.cleanup_semantic_pcd(pcd_pca, self.pcd_geo)
        if store_pcd: self.pcd_pca = pcd_pca
        self.publish_pointcloud(pcd_pca, self.pca_publisher, frame_id)

    def pub_cluster(self, model, store_pcd: bool = True, do_cleanup: bool = False, frame_id: str = "base_link") -> None:
        pcd_cluster = model.visualise_cluster_pcd()
        if do_cleanup: pcd_cluster = model.cleanup_semantic_pcd(pcd_cluster, self.pcd_geo)
        if store_pcd: self.pcd_cluster = pcd_cluster
        self.publish_pointcloud(pcd_cluster, self.cluster_publisher, frame_id)

    def pub_similarity(self, model, pos_prompts: List[str], neg_prompts: List[str] = [""], store_pcd: bool = True, 
                    do_cleanup: bool = True, frame_id: str = "base_link", threshold: float = -1000.) -> None:
        pcd_sim = model.query_similarity_pcd(pos_prompts, neg_prompts, threshold=threshold)
        if do_cleanup and self.pcd_geo is not None: pcd_sim = model.cleanup_semantic_pcd(pcd_sim, self.pcd_geo)
        if do_cleanup: pcd_sim = model.cleanup_geometric_pcd(pcd_sim)
        if store_pcd: self.pcd_sim = pcd_sim
        self.publish_pointcloud(pcd_sim, self.sim_publisher, frame_id)

    def pub_mask(self, model, pos_prompts: List[str], neg_prompts: List[str] = [""], store_pcd: bool = True, 
                    do_cleanup: bool = True, frame_id: str = "base_link", threshold: float = 0.5) -> None:
        pcd_mask = model.query_masked_pcd(pos_prompts, neg_prompts, threshold=threshold)
        #if do_cleanup: pcd_mask = model.cleanup_geometric_pcd(pcd_mask)
        if do_cleanup and self.pcd_geo is not None: pcd_mask = model.cleanup_semantic_pcd(pcd_mask, self.pcd_geo)
        if do_cleanup: pcd_mask = model.cleanup_geometric_pcd(pcd_mask)
        if store_pcd: self.pcd_mask = pcd_mask
        self.publish_pointcloud(pcd_mask, self.mask_publisher, frame_id)

    def republish_stored_pcds(self, frame_id: str = "base_link") -> None:
        if self.pcd_geo is not None: self.publish_pointcloud(self.pcd_geo, self.geo_publisher, frame_id)
        if self.pcd_pca is not None: self.publish_pointcloud(self.pcd_pca, self.pca_publisher, frame_id)
        if self.pcd_sim is not None: self.publish_pointcloud(self.pcd_sim, self.sim_publisher, frame_id)
        if self.pcd_mask is not None: self.publish_pointcloud(self.pcd_mask, self.mask_publisher, frame_id)
        if self.pcd_cluster is not None: self.publish_pointcloud(self.pcd_cluster, self.cluster_publisher, frame_id)

    def publish_pointcloud(self, pcd: o3d.geometry.PointCloud, publisher, frame_id: str = "base_link", convenience_rotate: bool = False) -> None:
        try:
            if convenience_rotate:
                pcd_out = deepcopy(pcd)
                R = pcd_out.get_rotation_matrix_from_xyz((np.pi/-2, 0, 0))  # -90 deg around x to turn in expected map orientation for ROS -> Convenient when initialising from VGGT
                pcd_out.rotate(R, center=(0,0,0))
                cloud_msg = pcl2roscloud(pcd_out, frame_id)
            else:
                cloud_msg = pcl2roscloud(pcd, frame_id)

            publisher.publish(cloud_msg)

            if self.automatic_spin: rclpy.spin_once(self, timeout_sec=0.1)

        except Exception as e:
            self.get_logger().error(f'Failed to publish pointcloud: {str(e)}')

##------------------
## Synchronises ROS query prompts on the feature field

class OTASQuerySubscriber(Node):
    def __init__(self, default_prompt: Optional[str] = None):
        super().__init__('otas_query_subscriber')
        self.query_sub = self.create_subscription(String, '/otas/query_prompt', self.query_callback, 1 )
        self._latest_query: Optional[str] = default_prompt
        self._trigger_query = False
        self._query_lock = threading.Lock()
            
    def query_callback(self, msg: String):
        with self._query_lock: 
            self._latest_query = msg.data
            self._trigger_query = True
    
    def get_latest_query(self) -> Optional[str]:
        """Atomically retrieves the latest query prompt."""
        with self._query_lock: 
            self._trigger_query = True
            return self._latest_query

    def get_trigger_query(self) -> bool:
        """Atomically retrieves the trigger query flag."""
        with self._query_lock: return self._trigger_query

    def set_trigger_query(self, trigger: bool = True) -> None:
        """Sets the trigger query flag."""
        with self._query_lock: self._trigger_query = trigger

##------------------
## Synchronises input RGB and depth streams for keyframe generation

def ros_image2pil(rgb_msg: ROSImage, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Converts ros image to pillow image for model inference. """
    rgb = _bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
    img = Image.fromarray(rgb)
    if target_size and (img.width, img.height) != target_size:
        img = img.resize(target_size, resample=Image.BILINEAR) # This should not be required, as we use the depth info for projection
    return img

def ros_depth2np(depth_msg: ROSImage) -> np.ndarray:
    """Converts metric depth ros image to numpy array for otas_spatial.reconstruct. Supports 32FC1, 32FC, 16UC1, MONO16 encodings."""
    arr = _bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    enc = depth_msg.encoding
    if enc == "32FC1" or enc == "32FC":
        return arr.astype(np.float32)                           # metres
    if enc == "16UC1" or enc == "MONO16":
        return (arr.astype(np.float32) / 1000.0)                # mm -> m
    # Some cameras publish "Type_32FC1" variants. Will add if needed
    raise ValueError(f"Unsupported depth encoding: {enc}")

def ros_image_calib2fu_fv_cu_cv(cam_info: CameraInfo) -> np.ndarray:
    """Converts camera calibration matrix to fu, fv, cu, cv for otas_spatial.reconstruct. Uses depth calibration for the 3D projection. """
    K = cam_info.k
    return np.array([K[0], K[4], K[2], K[5]], dtype=np.float64)

def tf_stamped_to_pose7(tf: TransformStamped) -> np.ndarray:
    """Converts ROS TransformStamped to 7D pose array for otas_spatial.reconstruct. The transform must be queried c2w (from camera optical_frame to world). Optical frame is in CV convention as typical in ROS 2. """
    # If cam driver does not publish an optical frame, maybe add a static transform here for convenience that rotates from ros' base frame in flu to the optical frame)
    t = tf.transform.translation
    q = tf.transform.rotation
    pose = np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w], dtype=np.float64)
    qn = np.linalg.norm(pose[3:7])
    if qn > 0:
        pose[3:7] /= qn
    return pose

class OTASInputSynchroniser(Node):
    def __init__(self, world_frame: str, camera_frame: str, rgb_topic: str, depth_topic: str, camera_info_topic: str):
        super().__init__('otas_input_synchroniser')
        self.world_frame  = world_frame
        self.camera_frame = camera_frame
        self.rgb_sub   = message_filters.Subscriber(self, ROSImage, rgb_topic, qos_profile=qos_profile_sensor_data) # By REP this profile should by used by cam drivers
        self.depth_sub = message_filters.Subscriber(self, ROSImage, depth_topic, qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=20, slop=0.08 )
        self.ts.registerCallback(self._sync_callback)

        ## Depth CameraInfo (thread-safe)
        self._depth_info_lock = threading.Lock()
        self._last_depth_info: Optional[CameraInfo] = None
        self.depth_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._depth_info_callback, 1 )

        ## Latest synchronised frame bundle (thread-safe)
        self._latest_frame_lock = threading.Lock()
        self._latest_frame: Optional[Dict[str, Any]] = None
        self._keyframe_requested: bool = True

        ## TF Buffering
        self._tf_node   = rclpy.create_node('otas_tf_buffer') # Spin as separate thread in the background to ensure up to date tfs -> This only works with the separate node, otherwise there will be a deadlock
        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._tf_node, spin_thread=True)

    def _depth_info_callback(self, msg: CameraInfo):
        with self._depth_info_lock: self._last_depth_info = msg

    def _get_latest_depth_info(self) -> Optional[CameraInfo]:
        with self._depth_info_lock: return self._last_depth_info

    def _get_transform(self, target_frame: str, source_frame: str, stamp) -> Optional[TransformStamped]:
        try:
            return self._tf_buffer.lookup_transform(target_frame, source_frame, Time.from_msg(stamp))
        
        except (TransformException, Exception) as e:
            self.get_logger().warn(f"TF lookup failed {source_frame}->{target_frame}: {e}")
            return None

    def _sync_callback(self, rgb_msg: ROSImage, depth_msg: ROSImage):
        """Converts ROS types to OTAS format and stores synchronised (rgb, depth, pose, calib, stamp) pair for the mapping server. """
        with self._latest_frame_lock: keyframe_requested = self._keyframe_requested

        if keyframe_requested:
            cam_info = self._get_latest_depth_info()
            if cam_info is None:
                self.get_logger().warn("No depth CameraInfo yet; skipping frame.")
                return

            depth_m = ros_depth2np(depth_msg)
            fu_fv_cu_cv = ros_image_calib2fu_fv_cu_cv(cam_info)
            rgb_pil = ros_image2pil(rgb_msg, target_size=(depth_msg.width, depth_msg.height))

            tf = self._get_transform(self.world_frame, self.camera_frame, depth_msg.header.stamp)
            if tf is None:
                self.get_logger().warn("No transform found for camera pose; skipping frame.")
                return

            pose7 = tf_stamped_to_pose7(tf)

            frame_bundle = {
                "rgb": rgb_pil,
                "depth": depth_m,
                "pose": pose7,
                "calib": fu_fv_cu_cv,
                "stamp": depth_msg.header.stamp, }

            # Make sure keyframe is still requested before storing
            with self._latest_frame_lock:
                if self._keyframe_requested:
                    self._latest_frame = frame_bundle
                    self._keyframe_requested = False

    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Atomically retrieves the latest RGB-D-Pose-Calib bundle. Returns None if no frame is available. """
        with self._latest_frame_lock:               
            bundle = self._latest_frame
            self._latest_frame = None
            return bundle

    def is_keyframe_available(self) -> bool:
        """Checks if a keyframe is available. """
        with self._latest_frame_lock: return self._latest_frame is not None

    def set_keyframe_requested(self, requested: bool = True) -> None:
        """Requests a synchronised and converted keyframe from the input stream. """
        with self._latest_frame_lock: self._keyframe_requested = requested

##------------------
## Stores frame data for reconstruction, tracks changes and checks validity of new frames

class OTASFrameBuffer:
    def __init__(self, max_len: int = 0, trunc_strategy: str = "drop_oldest", distance_threshold: float = 1.0, rotation_threshold: float = 0.8,
                        trans_velocity_threshold: float = 0.4, rotation_velocity_threshold: float = 0.4):
        
        assert trunc_strategy in ["drop_oldest", "equal_spacing"], f"Invalid truncation strategy: {trunc_strategy}. Must be 'drop_oldest' or 'equal_spacing'. Set max_len to 0 for unlimited length."
        self.depths = []
        self.imgs = []
        self.cam_poses = []
        self.fu_fv_cu_cvs = []
        self.stamps = []
        
        self.changed = False
        self.max_len = max_len
        self.trunc_strategy = trunc_strategy
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.trans_velocity_threshold = trans_velocity_threshold
        self.rotation_velocity_threshold = rotation_velocity_threshold
        self.buf_lock = threading.Lock()

    def add(self, frame_bundle: Dict[str, Any]) -> None:

        if len(self.cam_poses) > 0: # Always add first frame
            pose = frame_bundle["pose"]
            stamp = frame_bundle["stamp"]
            dtrans, drot = self.pose_delta(self.cam_poses[-1], pose)
            if dtrans < self.distance_threshold and drot < self.rotation_threshold: return # Skip if too close or rotated too little

            trans_vel, rot_vel = self.compute_velocity(self.cam_poses[-1], pose, self.stamps[-1], stamp)
            if trans_vel > self.trans_velocity_threshold or rot_vel > self.rotation_velocity_threshold: return # Skip if moving too fast (due to pose estim lag)
        
        with self.buf_lock:
            self.depths.append(frame_bundle["depth"])
            self.imgs.append(frame_bundle["rgb"])
            self.cam_poses.append(frame_bundle["pose"])
            self.fu_fv_cu_cvs.append(frame_bundle["calib"])
            self.stamps.append(frame_bundle["stamp"])

            # Limit number of frames either by dropping the oldest one or by equally spacing (200 imgs should be the limit for 8gb gpu)
            if self.max_len > 0 and len(self.depths) > self.max_len:
                if self.trunc_strategy == "drop_oldest":
                    self.depths.pop(0); self.imgs.pop(0); self.cam_poses.pop(0); self.fu_fv_cu_cvs.pop(0); self.stamps.pop(0)
                elif self.trunc_strategy == "equal_spacing":
                    self.depths = equally_space(self.depths, self.max_len)
                    self.imgs = equally_space(self.imgs, self.max_len)
                    self.cam_poses = equally_space(self.cam_poses, self.max_len)
                    self.fu_fv_cu_cvs = equally_space(self.fu_fv_cu_cvs, self.max_len)
                    self.stamps = equally_space(self.stamps, self.max_len)
            
            self.changed = True

    def get_buffers(self) -> Tuple[List[np.ndarray], List[Image.Image], List[np.ndarray], List[np.ndarray]]:
        """Getter for the full buffers for reconstruction. """
        # TODO: Clean the buffers using covisibility -> Probably by removing poses that are too close to each other (globally)
        # This would probably require a full graph though due to the combinatorial explosion of combinations
        # Maybe we need to accept this as a limit of the non-SLAM nature of this method
        with self.buf_lock:
            self.changed = False
            return self.depths, self.imgs, self.cam_poses, self.fu_fv_cu_cvs

    def get_changed(self) -> bool:
        """Getter for the changed flag indicating new keyframe availability. """
        with self.buf_lock: return self.changed

    @staticmethod
    def pose_delta(pose_last: np.ndarray, pose_new: np.ndarray) -> Tuple[float, float]:
        """Computes the delta in translation and rotation between two poses. Poses are in format [tx,ty,tz,qx,qy,qz,qw]. """
        p_last = np.array(pose_last); p_new = np.array(pose_new)
        dtrans = np.linalg.norm(p_new[:3] - p_last[:3])
        # Rotation angle between quaternions
        dot = abs(np.dot(p_last[3:7], p_new[3:7]))
        dot = max(-1.0, min(1.0, dot))
        drot = np.arccos(dot) * 2.0 # Convert to radians
        return dtrans, drot

    @staticmethod
    def compute_velocity(last_pose: np.ndarray, new_pose: np.ndarray, last_stamp: Time, new_stamp: Time) -> Tuple[float, float]:
        """Computes maximum velocity between two poses. Poses are in format [tx,ty,tz,qx,qy,qz,qw]. """
        d_pose = new_pose - last_pose 
        d_time = (new_stamp.sec - last_stamp.sec) + (new_stamp.nanosec - last_stamp.nanosec) * 1e-9
        m_trans_vel = np.max(np.abs(d_pose[:3] / d_time))
        m_rot_vel = np.max(np.abs(d_pose[3:7] / d_time))
        return m_trans_vel, m_rot_vel

##------------------
## Main synchronised mapping thread

def main():
    ## Model inference
    model = spatial_inference(config_path="./model_config/OTAS_ros2.json")
    config = model.config

    ## ROS 2 setup
    try: rclpy.init()
    except RuntimeError: pass # Ignore already initialised rclpy (for notebook)
    vis = OTASVisualizationNode(automatic_spin=False)
    query_node = OTASQuerySubscriber(
        default_prompt=config.get("map_server_default_prompt", None)) # Allows to optionally set a default prompt without publishing to /otas/query_prompt
    sync_node = OTASInputSynchroniser(
        world_frame = config["ros_world_frame"],
        camera_frame = config["ros_camera_optical_frame"],
        rgb_topic = config["ros_rgb_topic"],
        depth_topic = config["ros_depth_topic"],
        camera_info_topic = config["ros_camera_info_topic"])

    executor = MultiThreadedExecutor() # Sync nodes with multi-threaded spinning
    executor.add_node(vis)
    executor.add_node(query_node)
    executor.add_node(sync_node)

    ## Buffers
    buffer = OTASFrameBuffer(max_len=config.get("spatial_batch_size", 0),
                            trunc_strategy=config.get("spatial_trunc_strategy", "equal_spacing"),
                            distance_threshold=config["ros_keyframe_distance_threshold"],
                            rotation_threshold=config["ros_keyframe_rotation_threshold"],
                            trans_velocity_threshold=config["ros_keyframe_trans_velocity_threshold"],
                            rotation_velocity_threshold=config["ros_keyframe_rotation_velocity_threshold"])

    reconstruction_ready = False # Makes sure to never query before the first keyframe has been processed
    loop_idx = 0
    cleanup_countdown = -1

    sleep_time = 1/float(config["map_server_publish_rate"]) # Only implicit sleep with node spinning
    ## This might help to use ros rates without deadlock: https://robotics.stackexchange.com/questions/96684/rate-and-sleep-function-in-rclpy-library-for-ros2

    while rclpy.ok():
        sync_node.set_keyframe_requested()

        executor.spin_once(timeout_sec=sleep_time)

        ## MAPPING
        if sync_node.is_keyframe_available():
            frame = sync_node.get_latest_frame()
            if frame != None: buffer.add(frame) # This should always be True but just to be safe

        if loop_idx%40 == 0 and buffer.get_changed():
            depths, imgs, cam_poses, fu_fv_cu_cvs = buffer.get_buffers()
            model.reconstruct(imgs, depths, cam_poses, fu_fv_cu_cvs=fu_fv_cu_cvs)
            if config["map_server_pub_geometry"]: vis.pub_geometric(model, do_cleanup=False, frame_id=config["ros_world_frame"]) ## This registers the pcd's in the visualiser for free republishing
            if config["map_server_pub_pca"]: vis.pub_pca(model, do_cleanup=False, frame_id=config["ros_world_frame"])
            if config["map_server_pub_cluster"]: vis.pub_cluster(model, do_cleanup=False, frame_id=config["ros_world_frame"])
            clear_cuda_cache() # Cleanup to avoid accumulation of old embeddings on gpu
            reconstruction_ready = True
            query_node.set_trigger_query(True)
            cleanup_countdown = 15

        elif cleanup_countdown == 0: # Only in idle
            if config["map_server_pub_geometry"]: vis.pub_geometric(model, do_cleanup=config["map_server_do_cleanup"], frame_id=config["ros_world_frame"])
            if config["map_server_pub_pca"]: vis.pub_pca(model, do_cleanup=config["map_server_do_cleanup"], frame_id=config["ros_world_frame"])
            if config["map_server_pub_cluster"]: vis.pub_cluster(model, do_cleanup=config["map_server_do_cleanup"], frame_id=config["ros_world_frame"])
            cleanup_countdown = -1
        
        else:
            vis.republish_stored_pcds(frame_id=config["ros_world_frame"]) # Regularly republish for smooth visualisation -> This is mostly free due to buffering

            if config["map_server_do_cleanup"] and cleanup_countdown > 0:
                cleanup_countdown -= 1 # Only do this when we already have been in idle for a while
            
            ## MAP QUERIES
            prompt = query_node.get_latest_query() ## Only perform model query if prompt has changed as this is quite intensive with big clouds
            if query_node.get_trigger_query() and config["map_server_pub_mask"] and reconstruction_ready:
                vis.pub_mask(model, [prompt], neg_prompts=config["map_server_neg_prompts"],
                            threshold=config["map_server_threshold"], do_cleanup=config["map_server_do_cleanup"], frame_id=config["ros_world_frame"])

            if query_node.get_trigger_query() and config["map_server_pub_similarity"] and reconstruction_ready:
                vis.pub_similarity(model, [prompt], neg_prompts=config["map_server_neg_prompts"],
                                do_cleanup=config["map_server_do_cleanup"], frame_id=config["ros_world_frame"])
                
            if reconstruction_ready: query_node.set_trigger_query(False)

        if loop_idx < 40: loop_idx += 1
        else: loop_idx = 0

if __name__ == "__main__":
    main()
