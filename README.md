# Norm-Depth-3DGS
## 1. Environment Setup

### Prerequisites
Ensure your system meets the following requirements before running the project:
- Python version: Python 3.8.19
- Other necessary libraries
- environment.yml

## 2.Input Data File Format
datafile-name/
|——depths
        |——000000.png.npy
        |——000001.png.npy
        |——...
        |——{:06}".format(n).png.npy    \n-th file
|——images
        |——000000.png
        |——000001.png
        |——...
        |——{:06}".format(n).png    \n-th file
|——mask
        |——000000.png.png
        |——000001.png.png
        |——...
        |——{:06}".format(n).png.png    \n-th file
|——norms
        |——000000.png.npy
        |——000001.png.npy
        |——...
        |——{:06}".format(n).png.npy    \n-th file
|——sparse
        |——0
            |——0
                |——cameras.txt
                |——images.txt
                |——points3D.txt
## 3.Training Command
- python train.py -s <DATA_PATH> --masks <MASK_PATH>
## 4.Visualization Command
- SIBR_gaussianViewer_app.exe -m <result_PATH>
