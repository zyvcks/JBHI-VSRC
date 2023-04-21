# Source code of VSRC

## Introduction

The source code includes training and inference procedures for a semi-supervised medical image segmentation method with **Voxel Stability and Reliability Constraints (VSRC)**. 

Semi-supervised learning is becoming an effective solution in medical image segmentation because annotations are costly and tedious to acquire. Methods based on the teacher-student model use consistency regularization and uncertainty estimation and have shown good potential in dealing with limited annotated data. Nevertheless, the existing teacher-student model is seriously limited by the exponential moving average algorithm, which leads to the optimization trap. Moreover, the classic uncertainty estimation method calculates the global uncertainty for images but does not consider local region-level uncertainty, which is unsuitable for medical images with blurry regions. Here, the Voxel Stability and Reliability Constraint (VSRC) model is proposed to address these issues. Specifically, the Voxel Stability Constraint (VSC) strategy is introduced to optimize parameters and exchange effective knowledge between two independent initialized models, which can break through the performance bottleneck and avoid model collapse. Moreover, a new uncertainty estimation strategy, the Voxel Reliability Constraint (VRC), is proposed for use in our semi-supervised model to consider the uncertainty at the local region level. We further extend our model to auxiliary tasks and propose a task-level consistency regularization with uncertainty estimation. Extensive experiments on two 3D medical image datasets demonstrate that our method outperforms other state-of-the-art semi-supervised medical image segmentation methods under limited supervision.

This method has been submitted to the IEEE Journal of Biomedical and Health Informatics with title "**Semi-Supervised Medical Image Segmentation with Voxel Stability and Reliability Constraints**" (JBHI-03096-2022).

## Requirements 
- Python 3.7+ (Python 2 is not supported)
- PyTorch 1.7+
- CUDA 9.0+

**Note:** It is recommended to install Python and the necessary environment via [Anaconda](https://www.anaconda.com/).

## Directory structure

- ``trainer`` includes ``base_trainer.py`` which provides the basic training and validation process of the model.

- ``datasets`` is used to store training and test sets.

- ``data_loaders`` is used to store data loaders for different datasets. To use your own dataset, write the corresponding Python scripts and add the basic information in ``data_loaders/__init__.py``.

- ``data_augmentation`` contains the scripts of pre-processing and data augmentation functions for datasets.

- ``models`` is used to store models or networks for semi-supervised medical image segmentation. Note that, if you want to use your own network, write an appropriate class being derived from the class ``BaseModel`` defined in ``models/BaseModelClass.py`` and add the basic information in ``models/__init__.py``.

- ``losses3D`` stores the scripts of loss functions designed for VSRC.

- ``utils`` contains the scripts of some tool functions.

- ``visual3D`` is used to store the codes for inference and visualization of the test set. For using your own dataset, please add the basic information in ``visual3D/viz.py``.

- ``works`` is the directory used to store model checkpoints. You can change parameters ``args.l_save`` and ``args.r_save`` in ``train_vsrc.py`` to modify the default storage directory.

- ``runs`` is used to store log information recorded by tensorboard during runtime. 

## Usage

#### Step 1. Create a virtual environment and activate it in Anaconda

```shell
conda create -n vsrc python=3.7 -y
conda activate vsrc
```

#### Step 2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/)

```shell
conda install pytorch=1.7 torchvision cudatoolkit=10.0 -c pytorch
```

#### Step 3. Install other dependencies

```shell
pip install -r requirements.txt
```

#### Step 4. Prepare datasets

- Download the [Atrial Segmentation Challenge dataset (LA2018)](https://www.cardiacatlas.org/atriaseg2018-challenge/), follow the data preparation instructions of [UA-MT](https://github.com/yulequan/UA-MT) to convert medical images into h5 files and get the training and test sets. Taking the dataset LA2018 as an example, the prepared data looks as follows:
```
    datasets/
        ├── LA2018
            ├── train
            │    ├── IMAGE_ID_1
            │    │      └── la_mri.h5
            │    ├── IMAGE_ID_2
            │    │      └── la_mri.h5
            │    └── ...
            └── val
                 ├── IMAGE_ID_3
                 │      └── la_mri.h5
                 ├── IMAGE_ID_4
                 │      └── la_mri.h5
                 └── ...
```
- Similarly, you can follow the data preparation steps above for your own datasets. We provide the code to convert medical images to h5 files in ``utils/nrrd2h5.py`` (refer to [UA-MT](https://github.com/yulequan/UA-MT/blob/master/code/dataloaders/la_heart_processing.py)). Once the data is prepared (converted to h5 files and divided into training and test sets), write a data loader in Python, place it in the subdirectory ``data_loaders`` and add the basic information of the dataset in ``data_loaders/__init__.py``. The corresponding source code for the dataset LA2018 is provided in the subdirectory ``data_loaders`` for reference. 

#### Step 5. Train and test by running ``train_vsrc.py``

- Train the model on the specified dataset. The optional parameter ``--pretrained`` provides the ability to load pre-trained models.

```bash
python train_vsrc.py -d <DATASET_NAME> -p <PATCH_SIZE> --train
```

- Test and inference. Modify the codes in ``visual3D/viz.py`` for your own dataset. 
```bash
python train_vsrc.py -d <DATASET_NAME> -p <PATCH_SIZE> --test --save_viz -tp <TEST_CKPT_PATH> 
```

``DATASET_NAME`` is a string that specifies what dataset should be trained on, e.g. ``LA2018``. ``PATCH_SIZE`` is used to specify the patch size used for data training, e.g. ``112 112 80`` which means the patch size $112\times 112\times 80$. ``TEST_CKPT_PATH`` is used to specify the path where the model will be loaded during the test phase. 

For example, you can run following scripts to train and test VSRC on the LA2018 dataset:

```bash
python train_vsrc.py -d LA2018 -p 112 112 80 --train
python train_vsrc.py -d LA2018 -p 112 112 80 --test -save_viz -tp ./works/DualModel/test_checkpoint/sdf_VNet_LA2018_20.pth 
```

Our pre-trained models for the LA2018 dataset under 20% and 10% supervision are provided in the model directory ``works/DualModel/test_checkpoint``.

## Acknowledgement

- Some of the source code is referenced from [UA-MT](https://github.com/yulequan/UA-MT) and [DTC](https://github.com/HiLab-git/DTC).
- More semi-supervised learning approaches for comparison can be found in [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). We appreciate that Dr. Xiangde Luo provides an efficient code base.
