# ToMo-UDA++
ToMo-UDA++:Unsupervised Domain Adaptation for Anatomical Structure Detection Using Enhanced Topology and Morphology Knowledge - *Under Review*

**Notice**: This code repository is currently under peer review for IJCV. Please treat this as a preliminary version.


## PostScript
 This project is the pytorch implemention of ToMo-UDA++;

 Our experimental platform is configured with <u>One *RTX3090 (cuda>=11.7)*</u>; 

 Currently, this code is avaliable for public dataset <a href="https://github.com/xmed-lab/GraphEcho">CardiacUDA</a>, <a href="https://github.com/xmed-lab/ToMo-UDA">FUSH</a> and our dataset <a href="https://drive.google.com/drive/folders/1pZ-B_Tnu2qnuYZKO1XDHG9dGe8BGyVX7?usp=drive_link">FUSSD<sup>3</sup></a>;  

 ## Installation

### Prerequisites

- Python ≥ 3.6
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.
- Detectron2 == 0.5

### Install python env

To install required dependencies on the virtual environment of the python (e.g., virtualenv for python3), please run the following command at the root of this code:
```
$ python3 -m venv /path/to/new/virtual/environment/.
$ source /path/to/new/virtual/environment/bin/activate
```
For example:
```
$ mkdir python_env
$ python3 -m venv python_env/
$ source python_env/bin/activate
```
 

### Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

## Dataset download

1. Download the datasets
   
   | FUSSD<sup>3</sup> Dataset       |          |
   |----------------------------------|----------|
   | **Annotated Images**             | 4,654    |
   | **Resolution**                   | 480-1080p|
   | **Views**                        | 1        |
   
   | **Structure**                    | **Abbreviation** |
   |----------------------------------|------------------|
   | Skin Contour                     | SC               |
   | Vertebral Arch Ossification Center | VAO           |
   | Medulla Spinalis                 | MS               |
   | Medullary Cone                   | MC               |
   | Vertebral Ossification Center    | VOC              |
   | Spinal End                       | SE               |
   
   <a href="https://drive.google.com/drive/folders/1pZ-B_Tnu2qnuYZKO1XDHG9dGe8BGyVX7?usp=drive_link">FUSSD<sup>3</sup></a> is a pure cross-device detection released dataset collected from Samsung (SA), Philips (PH), and GE devices with numbers of 1882, 1510, and 1262, respectively, and is labeled with 6 anatomy regions, which is the first benchmark for the structure detection of pure cross-multi-device.


3. Organize the dataset as the COCO annotation format.

## Training

- Train ToMo-UDA++ under SA (source) and GE (target) on FUSSD<sup>3</sup> dataset

```shell
python train_net.py \
      --num-gpus 1 \
      --config configs/frcnn_res50fpn_spine_sa_ge\
      OUTPUT_DIR output/fussd_sa_ge
```

## Resume the training

```shell
python train_net.py \
      --resume \
      --num-gpus 1 \
      --config configs/frcnn_res50fpn_spine_sa_ge MODEL.WEIGHTS <your weight>.pth
```

## Evaluation

Download the ```CHECKPOINT``` <a href="https://drive.google.com/drive/folders/1pZ-B_Tnu2qnuYZKO1XDHG9dGe8BGyVX7?usp=drive_link">here</a>.
```shell
python train_net.py \
      --eval-only \
      --num-gpus 1 \
      --config configs/test_res.yaml \
      MODEL.WEIGHTS <CHECKPOINT>.pth
```


## Important Notes for Reviewers
- This code is provided for review purposes only and may undergo further refinement before final publication.
- Some implementation details might be adjusted based on reviewer feedback.
- For any questions regarding the code or reproduction of results, please contact: [lvxg@stu.ahu.edu.cn](mailto:lvxg@stu.ahu.edu.cn).

