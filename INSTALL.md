## Set up the python environment

### conda environment
```
conda create -n sanerf python=3.7
conda activate sanerf
```

### install dependencies
```
# make sure that the pytorch cuda is consistent with the system cuda
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu101_pyt160/download.html

pip install -r requirements.txt

conda install -c conda-forge pyembree
```

## Set up the SMPL
Download the neutral `.pkl` model from SMPLify.
1. Register an account and download the SMPL model (`SMPLIFY_CODE_V2.ZIP`) from [here](https://smplify.is.tue.mpg.de/index.html).
2. Find `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`, rename it to `SMPL_NEUTRAL.pkl` and put it to `$ROOT/zju_smpl/smplx/smpl/SMPL_NEUTRAL.pkl`.

Download the `.obj` model from SMPL.
1. Register an account and download the SMPL model (`Download UV map in OBJ format`) from [here](https://smpl.is.tue.mpg.de/index.html).
2. Find `smpl_uv.obj` and put it to `$ROOT/zju_smpl/smplx/smpl/smpl_uv.obj`.

## Set up datasets

### ZJU-Mocap dataset

1. Follow the instructions [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset) to download the ZJU-Mocap dataset.
2. Create a soft link:
    ```
    ROOT=/path/to/sanerf
    cd $ROOT/data
    ln -s /path/to/zju_mocap zju_mocap
    ```