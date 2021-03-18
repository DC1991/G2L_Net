# 6D Pose Estimation
## Experiment setup

OS: Ubuntu 16.04

GPU: 1080 Ti

Programme language: Python 3.6


## Architecture
Our framework is built on [Frustum-Pointnet](https://github.com/charlesq34/frustum-pointnets), but we reimplemented the code to pytorch (we borrom some code from this [git](https://github.com/fxia22/pointnet.pytorch)) with some improvements (contributions):   
1. First, instead of locating the object point cloud by a frustum, we locate the object point cloud by a 3D sphere, which can limit the 3D search range in a more compact space.   
2. Second, instead of directly regressing the global point feature to estimate the pose, we propose the point-wise embedding vector features to effectively capture the viewpoint information.  
3. Third, we estimate the rotation residual between predicted rotation and the ground truth. The rotation residual estimator further boosts the pose estimation accuracy.

More details can be found in our [paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_G2L-Net_Global_to_Local_Network_for_Real-Time_6D_Pose_Estimation_CVPR_2020_paper.pdf). Our AD_VIDEO for CVPR 2020 Virtual Presentation is availabel at [youtube](https://www.youtube.com/watch?v=a5JWe6mOAEs).

If you find our code or paper is useful, please cite our paper:


@InProceedings{Chen_2020_CVPR,<br>
author = {Chen, Wei and Jia, Xi and Chang, Hyung Jin and Duan, Jinming and Leonardis, Ales},<br>
title = {G2L-Net: Global to Local Network for Real-Time 6D Pose Estimation With Embedding Vector Features},<br>
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},<br>
month = {June},<br>
year = {2020}<br>
}<br>

If you also interesting in how to do category-level pose estimation please refer to our new [work](https://github.com/DC1991/FS-Net).

## Pre requirements
Install the libraries by:

python install -r requirements.txt

The trained models can be download from [here](https://drive.google.com/open?id=1aOyaUf07-UZRIDvdX3XJglhrnVq8qbzu).

## Demo & Testing
The command for demo:

cd demo/

python test_linemod

Please note, we only provide 20 images for quick test, if you need to test the whole sequence, you can download the LINEMOD dataset [here](http://ptak.felk.cvut.cz/6DB/public/bop_datasets/lm_test_all.zip).

## Training

Due to some minor changes to the code, the parameters are slightly different from the paper.

### YOLOv3
For implementation & training YOLOv3 please refer to this [git](https://github.com/ultralytics/yolov3).

### G2L_Net

#### training data
We provide the point cloud (and point wise label) which transfered from depth image of object 'cat' in LINEMOD. Please download the training data from [here](https://drive.google.com/drive/folders/194BceEL-YV6MXW9v0iE8uau_ZzFvN2RL?usp=sharing). After downloading the data, please unzip the data to "train_data" folder. Please note, in the data the point cloud has six dimension:[x, y, z, r, g, b], we only use the first three dimension:[x, y, z]. For the corresponding label, it has 13 dimension, we only use the first dimension. The last 12 values are useless now. 

#### training command
cd train/

python train_G2L.py




