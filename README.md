# A hybrid neural network for mangrove mapping considering tide states using Sentinel-2 imagery
Author: Longjie Ye, Qihao Weng* | [link](https://www.sciencedirect.com/science/article/pii/S0034425725003219)
Remote Sensing of Environment

## Introduction
Existing machine learning algorithms for mangrove mapping are incapable of delivering precise cartographic solutions under dynamic tidal conditions because of poor transferability. This study developed a generalized approach for large-area mangrove mapping using a hybrid neural network integrated with a vision transformer to effectively capture representative features. This project used Sentinel-2 imagery together with high-resolution Google Earth & UAV imagery to map the mangrove extent at a resolution of 10 meters for 2021. 

This project was built upon the Python & Pytorch environment. To reduce pre-training time, we used the model weights pre-trained on open-source datasets provided by the segmentation_models.pytorch library.
 

## Set up environment
Before going deeper, we need to set up the code environment. We encourage you to follow the guidelines below.
```bash
conda create -n hybridNet python=3.7
conda activate hybridNet

conda install pytorch=1.12.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia

python -m pip install opencv-python-headless==4.6.0.66

pip install -U albumentations --no-binary qudida,albumentations

#This file can be found in this project
pip install -r requirements.txt

```

## Data usage
The original Sentinel-2 images were downloaded from the Google Earth Engine. Make sure you have registered for one account in [GEE](https://earthengine.google.com).

## Getting started with code
To reproduce the results presented in our paper, you need to create wandb account and specify your own wandb account name in configs/*.yaml. 
```bash
# For training
python train.py

```

## TO BE ADDED
The pretrained .pth will be released soon.


# Citing
If you find our work useful, please do not hesitate to cite it
```
{Longjie Ye, Qihao Weng,
A hybrid neural network for mangrove mapping considering tide states using Sentinel-2 imagery,
Remote Sensing of Environment,
Volume 329,
2025,
114917,
ISSN 0034-4257,
https://doi.org/10.1016/j.rse.2025.114917}
```
[HybridNet](https://www.sciencedirect.com/science/article/pii/S0034425725003219)





