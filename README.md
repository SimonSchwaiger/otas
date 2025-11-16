<h1 align="center">
OTAS: Open-vocabulary Token Alignment for Outdoor Segmentation
</h1>

<h3 align="center">
Simon Schwaiger<sup>1,2</sup>, Stefan Thalhammer<sup>2</sup>, Wilfried Wöber<sup>2,3</sup> and Gerald Steinbauer-Wagner<sup>1</sup>
</h3>

<p align="center">
<sup>1</sup>Graz University of Technology<br>
<sup>2</sup>University of Applied Sciences Technikum Wien<br>
<sup>3</sup>University of Natural Resources and Life Sciences Vienna
</p>

<p align="center">
🌐 <a href="https://otas-segmentation.github.io">otas-segmentation.github.io</a>
</p>

***************************************

<div style="max-width: 800px; margin: auto;", align="center">
  <img src="./img/otas_contrib_demo.gif" alt="OTAS Capability Overview GIF" width="48%">
  <img src="./img/otas_experiments_demo.gif" alt="OTAS Semantic Mapping Demo GIF" width="48%">
</div>

***************************************

## 🚀 Getting Started

OTAS is an open-vocabulary segmentation and semantic reconstruction model that aligns foundation model output tokens across one or multiple input views without any training. This results in a parameter-efficient and lightweight language embedding approach, especially effective for outdoor tasks. Ready to dive in?

**Prerequisites**

1. 📂 Clone this repository: `git clone --recursive https://github.com/SimonSchwaiger/otas`
2. 📦 Install dependencies: `pip install -r requirements.txt`
3. ⬇️ Make sure you have wget installed and run the `download_checkpoints.sh` script
4. 🎯 Import inference helper from `src/inference.py`. See `demo.ipynb` for usage!

**📁 Repository Structure**

* 🔧 `src/inference.py` - Your main inference helper functions (check out `demo.ipynb` for examples!)
* 🧠 `src/model.py` - The core model with language embedding and reconstruction
* ⚙️ `src/model_config.py` - Model configuration loader (you can create custom configs as JSON files and point to them with `OTAS_CONFIG_PATH`)

**🎮 Inference Helper**

We include a demo notebook that shows how to use the convenience wrappers for easy inference. You can even import from and export to Nerfstudio datasets for easy reconstruction! See the notebook at [`./demo.ipynb`](./demo.ipynb).

***************************************

## 🏔️ Few-Line Environment Reconstruction Example

With the repository downloaded, inference just takes a few lines of code by initialising OTAS from [VGGT](https://github.com/facebookresearch/VGGT). Here is a reconstruction of a hiking trail in the alps.

The input images look something like this:

<center><img src="./img/demo/dataset_demo.png" alt="Example Image from Dataset" width="250"></center>

Using OTAS, a semantic map can easily be reconstructed, allowing for open-vocabulary queries! 🗺️

```python
import open3d as o3d
import sys; sys.path.append("otas/src")
from inference import spatial_inference, all_imgs_in_dir

img_paths = all_imgs_in_dir("./img/outdoor_reconstruction")
model = spatial_inference(vggt_image_paths = img_paths)

## Geometry
pcd = model.cleanup_overexposed_pcd(model.pcd_colour)
o3d.io.write_point_cloud("./otas_reconstruction_geometric.ply", pcd)

## PCA
pcd_pca = model.visualise_pca_pcd()
o3d.io.write_point_cloud("./otas_reconstruction_pca.ply", pcd_pca)

## Open-Vocabulary Query
pcd = model.query_relevance_pcd(["wooden bridge"], ["object"])
o3d.io.write_point_cloud("./otas_reconstruction_similarity.ply", pcd)
```

Here are the results! 🎉 The following images are a top-view over the geometric reconstruction (left), PCA over language embeddings (middle), and semantic similarity to the prompt _**"wooden bridge"**_ (right):

<div style="max-width: 800px; margin: auto;", align="center">
  <img src="./img/demo/otas_spatial_demo_colour.png" alt="Geometric Reconstruction Demo" width="31%">
  <img src="./img/demo/otas_spatial_demo_pca.png" alt="PCA over Semantics Demo" width="31%">
  <img src="./img/demo/otas_spatial_demo_similarity.png" alt="Similarity to Prompt Demo" width="31%">
  </div>

***************************************

## 🤖 Real-Time ROS 2 Integration

The included ROS 2 node in `src/ros2_node.py` allows for real-time semantic mapping and simultaneous open-vocabulary queries. It requires RGB, depth and camera info topics to be published. Camera poses are tracked from TF.

1. Install ROS 2 (tested with Humble)
2. Set up OTAS in a virtual environment with dependencies installed and checkpoints downloaded (see Getting Started)
3. Install ROS 2 dependencies: `apt update && apt install -y ros-$ROS_DISTRO-sensor-msgs-py ros-$ROS_DISTRO-message-filters ros-$ROS_DISTRO-cv-bridge`. Depending on the provided message-filters and cv-bridge binaries, you may need to downgrade numpy: `pip install "numpy<2"`
4. Adjust configuration file (e.g. `src/model_config/OTAS_ros2.json`) to your sensor setup. The following settings must match your ROS 2 environment:
  * `ros_world_frame` - World coordinate frame name (will be the common frame for all published pointclouds). If you run a SLAM alongside to track camera poses, make sure that TF updates from this (world) frame to the camera optical frame are time-synchronized with the camera streams.
  * `ros_camera_optical_frame` - Camera *optical* frame. Make sure that (matching ROS 2 conventions) this frame follows the OpenCV convention of z-forward, y-down, x-right. Most camera drivers publish a static transform from the camera base frame to this optical frame.
  * `ros_rgb_topic` - RGB image stream
  * `ros_depth_topic` - Depth image stream
  * `ros_camera_info_topic` - Camera info topic (use the depth camera's info here as published by the depth camera driver)
5. Run the ROS 2 node: `python ros2_node.py`

**Published Topics**

The ROS 2 node publishes the following topics. All are of type Pointcloud2 and in the world frame:

* `/otas/pointcloud/geometric` - Geometric pointcloud
* `/otas/pointcloud/pca` - PCA over language embeddings
* `/otas/pointcloud/similarity` - Semantic similarity to a given prompt
* `/otas/pointcloud/mask` - Semantic segmentation mask
* `/otas/pointcloud/cluster` - Semantic clusters

**Subscribed Topics**

* `/otas/query` - Query prompt to trigger publishing of similarity and mask pointclouds

**Parameters**

* `/otas/save_map_path` - Path to save the map to disk. Set using `ros2 param set /otas_frame_buffer otas_save_map_path "<path>"`

**Services**

* `/otas/save_map` - Save the current map to disk in Nerfstudio format. Saves by default to `<otas_dir>/saved_maps/<timestamp>`. Save location can be changed by setting the `/otas/save_map_path` parameter. Call using `ros2 service call otas/save_map std_srvs/srv/Trigger`
* `/otas/clear_map` - Clear the mapped keyframes. Call using `ros2 service call otas/clear_map std_srvs/srv/Trigger`

***************************************

## 📄 Citation

If you use this work in your research, please cite our paper:

```bibtex
@misc{Schwaiger2025OTAS,
    title               = {OTAS: Open-vocabulary Token Alignment for Outdoor Segmentation. \textit{arXiv preprint arXiv:2507.08851}}, 
    author              = {Simon Schwaiger and Stefan Thalhammer and Wilfried Wöber and Gerald Steinbauer-Wagner},
    year                = {2025},
    url                 = {https://arxiv.org/abs/2507.08851}
}
```

***************************************

## 🙏 Acknowledgement

Thanks to these repositories and works! Without them, this research wouldn't have been possible:

* [DINOv2](https://github.com/facebookresearch/dinov2)
* [Maskclip_onnx](https://github.com/RogerQi/maskclip_onnx)
* [CLIP](https://github.com/openai/CLIP)
* [Open3D](https://github.com/isl-org/Open3D)
* [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
* [VGGT](https://github.com/facebookresearch/VGGT)
* [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
* [AM-RADIO](https://github.com/NVlabs/RADIO)