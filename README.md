# 3DMM model fitting using Pytorch

This is a fitting framework implemented in Pytorch for reconstructing faces from images using BFM. 

The frame only uses Pytorch modules and a differentiable renderer from pytorch3d. The whole module is differentiable and can be integrated into other systems for the gradient propagation. 

<p align="center">
  <img src="gifs/demo.gif" alt="demo" width="512px">
</p>

## Installation
### Requirements
- [pytorch3d](https://github.com/facebookresearch/pytorch3d) It might require a specific version of Pytorch to make pytorch3d run succussfully on gpus, please follow the official instructions.
- Please refer to "requirements.txt" for other dependences.
- [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model)
- [Expression Basis](https://github.com/Juyong/3DFace) extra expression basis.

## Instruction
1. Clone the repo:
```
git clone https://github.com/ascust/3DMM-Fitting-Pytorch
cd 3DMM-Fitting-Pytorch
```

2. Download the Basel Face Model and put "01_MorphableModel.mat" and put it into "BFM".

3. Download the Expression Basis. Go to the [repo](https://github.com/Juyong/3DFace), download the "CoarseData" and put "Exp_Pca.bin" into "BFM".

4. Convert the BMF parameters by:
```
python convert_bfm_data.py 
```

5. Run the code on specific images by:
```
python fit.py --img data/000002.jpg
```
The code will do rigid fitting as well as non-rigid fitting using landmarks as well as the image as supervision. 


## Acknowledgement
The code is partially borrowed from [Deep3DFaceReconstrution](https://github.com/microsoft/Deep3DFaceReconstruction), which is a Tensorflow-based deep reconstruction method using CNNs. Please note that our framework does not require any pretrained deep models. We estimate the parameters directly using the landmarks and photometric loss as the supervision.