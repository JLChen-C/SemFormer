# SemFormer
The official code for `SemFormer: Semantic Guided Activation Transformer for Weakly Supervised Semantic Segmentation`.

# Runtime Environment
- Python 3.6
- PyTorch 1.7.1
- CUDA 11.0
- 2 x NVIDIA A100 GPUs
- more in requirements.txt

# Usage

## Install python dependencies
```bash
python -m pip install -r requirements.txt
```

## Download PASCAL VOC 2012 devkit
Follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

## Train and evaluate the model.

### 1. Train Semformer for generating CAMs
1.1 Train CAAE.
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_caae.py --tag CAAE@DeiT-B-Dist
```
1.2 Train Semformer.
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_semformer.py --tag SemFormer@CAAE@DeiT-B-Dist
```

### 2. Inference SemFormer for generating CAMs
```bash
CUDA_VISIBLE_DEVICES=0 python inference_semformer.py --tag SemFormer@CAAE@DeiT-B-Dist --domain train_aug
```
Evaluate CAMs. [optinal]
```bash
python evaluate.py --experiment_name SemFormer@CAAE@DeiT-B-Dist@train@scale=0.5,1.0,1.5,2.0 --domain train
```

### 3. Apply Random Walk (RW) to refine the generated CAMs
2.1. Make affinity labels to train AffinityNet.
```bash
python make_affinity_labels.py --experiment_name SemFormer@CAAE@DeiT-B-Dist@train@scale=0.5,1.0,1.5,2.0 --domain train_aug
```

2.2. Train AffinityNet using the generated affinity labels.
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_affinitynet.py --tag AffinityNet@SemFormer --label_name SemFormer@CAAE@DeiT-B-Dist@train@scale=0.5,1.0,1.5,2.0@aff_fg=0.11_bg=0.15
```

### 4. Train and Evaluate the segmentation model using the pseudo-labels
Please follow the instructions in [this repo](https://github.com/YudeWang/semantic-segmentation-codebase) to train and evaluate the segmentation model.

### 5. Results
Qualitative segmentation results on PASCAL VOC 2012 (mIoU (%)). Supervision: pixel-level (F), box-level (B), saliency-level (S), and image-level (I).

|---------|---------------|---------------|---------|----------|
| Method  |  Publication  |  Supervision  |  *val*  |  *test*  |
|---------|:-------------:|:-------------:|:-------:|:--------:|
|DeepLabV1|[ICLR'15](https://arxiv.org/abs/1412.7062)|F|68.7|71.6|
|DeepLabV2|[TPAMI'18](https://arxiv.org/abs/1606.00915)|F|77.7|79.7|
|---------|---------------|---------------|---------|----------|
|BCM|[CVPR'19](https://arxiv.org/abs/1904.11693)|I + B|70.2|-|
|BBAM|[CVPR'21](https://arxiv.org/abs/2103.08907)|I + B|73.7| 73.7|
|---------|---------------|---------------|---------|----------|
|ICD|[CVPR'20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Learning_Integral_Objects_With_Intra-Class_Discriminator_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2020_paper.pdf)|I + S|67.8|68.0|
|EPS|[CVPR'21](https://arxiv.org/abs/2105.08965)|I + S|71.0|71.8|
|---------|---------------|---------------|---------|----------|
|BES|[ECCV'20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710341.pdf)|I|65.7|66.6|
|CONTA|[NeurIPS'20](https://arxiv.org/abs/2009.12547)|I|66.1|66.7|
|AdvCAM|[CVPR'21](https://arxiv.org/abs/2103.08896)|I|68.1|68.0|
|OC-CSE|[ICCV'21](https://openaccess.thecvf.com/content/ICCV2021/papers/Kweon_Unlocking_the_Potential_of_Ordinary_Classifier_Class-Specific_Adversarial_Erasing_Framework_ICCV_2021_paper.pdf)|I|68.4|68.2|
|RIB|[NeurIPS'21](https://arxiv.org/abs/2110.06530)|I|68.3|68.6|
|CLIMS|[CVPR'22](https://arxiv.org/abs/2203.02668)|I|70.4|70.0|
|MCTFormer|[CVPR'22](https://arxiv.org/abs/2203.02891)|I|71.9|71.6|
|**SemFormer (ours)**| - |I|**73.7**|**73.2**|
|---------|---------------|---------------|---------|----------|

# Acknowledgement
This repo is modified from [Puzzle-CAM](https://github.com/OFRIN/PuzzleCAM), thanks for their contribution to the community.