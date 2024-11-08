# [NeurIPS'24] Continuous Heatmap Regression for Pose Estimation via Implicit Neural Representation
![xxx](figures/overview.svg)
By introducing implicit neural representation (INR), the proposed NerPE achieves continuous heatmap regression for 2D human pose estimation. Thanks to the decoupling of INR and spatial resolution, NerPE can output the predicted heatmaps at arbitrary resolutions. See Continuous Heatmap Regression for Pose Estimation via Implicit Neural Representation for details.

## Implementation
The code for NerPE is being sorted out and coming soon.

<!--

## Installation & Preparation

Please replace the values ​​of DATASET.ROOT, MODEL.PRETRAINED, and TEST.COCO_BBOX_FILE in the configuration file with the corresponding paths. The details of dependency installation and data preparation are given [here](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/). 

## Training & Testing
**Training on MPII dataset**
```
python tools/train.py \
    --cfg experiments/mpii/hrnet/w32_128x128_feat_8x8_gau_0.06.yaml
```
**Testing on MPII dataset**
```
python tools/test.py \
    --cfg experiments/mpii/hrnet/w32_128x128_feat_8x8_gau_0.06.yaml \
    TEST.MODEL_FILE _PATH_TO_CHECKPOINT_
```
The commands used for COCO and CrowdPose are similar, where the configuration file is changed to the corresponding one and TEST.USE_GT_BBOX is set to false during testing.

**Note:**
- There may be rare problems after the code is sorted out. If this happens, please contact us.
- NerPE-p, which uses progressive coordinate decoding, will be integrated into the project later.

## Citation
If the project helps your research, please consider citing our papers: 
```
@inproceedings{Hu2024Continuous,
    author    = {Shengxiang Hu, Huaijiang Sun, Dong Wei, Xiaoning Sun, Jin Wang},
    title     = {Continuous Heatmap Regression for Pose Estimation via Implicit Neural Representation},
    booktitle = {Advances in Neural Information Processing Systems(NeurIPS)},
    year      = {2024}
}
```

-->

## Acknowledgement
This project is developed based on [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/) and [LIIF](https://github.com/yinboc/liif).
