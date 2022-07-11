# Surface-Aligned Neural Radiance Fields for Controllable 3D Human Synthesis (CVPR 2022)

### [Project page](https://pfnet-research.github.io/surface-aligned-nerf/) | [Video](https://youtu.be/cD3hMbkFk9Y) | [Paper](https://arxiv.org/pdf/2201.01683.pdf)

![pipeline](https://pfnet-research.github.io/surface-aligned-nerf/files/pipeline.png)

## Installation and Data Preparation

Please see [INSTALL.md](INSTALL.md) for installation and data preparation.

## Run the code on ZJU-MoCap
Take the subject "313" as an example.

### Download the pre-trained model
For a quick start, you can download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1K-sTF26We3xC6Z1qrXn3PCd-DRTve1kc?usp=sharing) **(to be updated)**, and put it to `$ROOT/data/trained_model/sa-nerf/test_313/latest.pth`.

### Evaluation
Test on training human poses:
```
python run.py --type evaluate --cfg_file configs/zju_mocap_exp/multi_view_313.yaml exp_name test_313
```
Test on unseen human poses:
```
python run.py --type evaluate --cfg_file configs/zju_mocap_exp/multi_view_313.yaml exp_name test_313 test_novel_pose True
```
Then you can get a quantitative evaluation in terms of PSNR and SSIM. (The results may slightly differ from those reported in the paper, because we found that we originally used [the incorrect ray sampling code](https://github.com/zju3dv/neuralbody/commit/c47c36554c2991c4cbaf2a370c42fdee51bfb451) in the Neural Body pipeline. Here we use the updated code.)

### Novel view synthesis (rotate camera)

<img src="https://pfnet-research.github.io/surface-aligned-nerf/files/rotate.gif" width="30%">

```
python run.py --type visualize --cfg_file configs/zju_mocap_exp/multi_view_313.yaml exp_name test_313 vis_novel_view True
```

### Shape control (change SMPL shape parameters)

<img src="https://pfnet-research.github.io/surface-aligned-nerf/files/shape_control.gif" width="30%">

Here we use the list of `[pc, val]` to specify the principal components of SMPL. 
Here, `pc` represents the number of principal components and `val` represents the value of principal components. 
For example, if you want to set `pc1` to `4` and `pc2` to `-2`, run:
```
python run.py --type visualize --cfg_file configs/zju_mocap_exp/multi_view_313.yaml exp_name test_313 vis_novel_view True shape_control "[[1, 4], [2, -2]]"
```

###  Changing clothes

<img src="https://pfnet-research.github.io/surface-aligned-nerf/files/changing_clothes.gif" width="30%">

For example, if you want to use the upper body of subject "377" and lower body of subject "390" for rendering, run:
```
python run.py --type visualize --cfg_file configs/zju_mocap_exp/multi_view_313.yaml exp_name test_313 vis_novel_view True upper_body test_313 lower_body test_377
```
You may need to adjust the threshold [here](lib/networks/renderer/if_clight_renderer.py#L77) for a better segmentation.

### Training on ZJU-MoCap from scratch
```
python train_net.py --cfg_file configs/zju_mocap_exp/multi_view_313.yaml exp_name test_313 resume False
```

## Implementation of the dispersed projection
If you want to learn more about the implementation details of the proposed dispersed projection, check [here](lib/networks/projection/map.py).

## Citation
```bibtex
@inproceedings{xu2022sanerf,
    author    = {Xu, Tianhan and Fujita, Yasuhiro and Matsumoto, Eiichi},
    title     = {Surface-Aligned Neural Radiance Fields for Controllable 3D Human Synthesis},
    booktitle = {CVPR},
    year      = {2022},
}
```

## Acknowledgement

Our implementation is based on the code for the [Neural Body (Peng et al., CVPR 2021)](https://zju3dv.github.io/neuralbody/).
We thank the authors for releasing the code, and please also consider citing their work.