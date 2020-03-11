Code will be release soon.

# Experiment setup

OS: Ubuntu 16.04

GPU: 1080 Ti

Programme language: Pytorch 1.0


# Architecture
Our framework is built on [Frustum-Pointnet](https://github.com/charlesq34/frustum-pointnets), but we reimplemented the code to pytorch (we borrom some code from this [git](https://github.com/fxia22/pointnet.pytorch).) with some improvements (contributions): 
#### First, instead of locating the object point cloud by a frustum, we locate the object point cloud by a 3D sphere, which can limit the 3D search range in a more compact space. 
#### Second, instead of directly regressing the global point feature to estimate the pose, we propose the point-wise embedding vector features to effectively capture the viewpoint information. 
#### Third, we estimate the rotation residual between predicted rotation and the ground truth. The rotation residual estimator further boosts the pose estimation accuracy.
