## *SegGCN: Efficient 3D Point Cloud Segmentation with Fuzzy Spherical Kernel*
Created by Huan Lei, Naveed Akhtar and Ajmal Mian

### Introduction
We introduce the concept of fuzzy kernel in this work and present a scalable network architecture which is effective to process **1M+** points per second in the inference stage. 

Fuzzy clustering is known to perform well in real-world applications. Inspired by this observation, we incorporate a fuzzy mechanism into discrete convolutional kernels for 3D point clouds as our first major contribution. The proposed fuzzy  kernel is defined over a spherical volume that uses discrete bins. Discrete volumetric division can normally make a kernel vulnerable to boundary effects during learning as well as point density during inference. However, the proposed kernel remains robust to boundary  conditions and point density due to the fuzzy mechanism. Our second major contribution comes as the proposal of an efficient graph convolutional network, SegGCN for segmenting point clouds. The proposed network exploits ResNet like blocks in the encoder and $1\times1$ convolutions in the decoder. 
SegGCN capitalizes on the separable convolution operation of the proposed fuzzy kernel for efficiency. We establish the effectiveness of the SegGCN with the proposed kernel on the challenging S3DIS and ScanNet real-world datasets. Our experiments demonstrate that the proposed network can segment **over one million points per second** with highly competitive performance. 

In this repository, we release the code and trained models for segmentation.

### Citation
If you find our work useful in your research, please consider citing:

```
@article{lei2019spherical,  
  title={SegGCN: Efficient 3D Point Cloud Segmentation with Fuzzy Spherical Kernel},  
  author={Lei, Huan and Akhtar, Naveed and Mian, Ajmal},  
  journal={IEEE Conference on Computer Vision and Pattern Recognition},  
  year={2020}  
}
```

