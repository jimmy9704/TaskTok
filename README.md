# TaskTok: Delving into Task Tokens for Task-driven Image Restoration

<p align="center">
  <img src="tasktok_logo.png" alt="tasktok_logo" width="300">
</p>

This repository contains the official implementation of our paper ["TaskTok: Delving into Task Tokens for Task-driven Image Restoration"](https://arxiv.org/abs/2606.26615).

Our implementation is based on [EDTR](https://github.com/JaehaKim97/EDTR) and [TiTok](https://github.com/bytedance/1d-tokenizer).

### 🛠️ Conda Environment Setup

```shell
conda create -n tasktok python=3.10
conda activate tasktok
pip install -r requirements.txt
```

### ⚡ Quick Start

You can quickly test TaskTok using the sample images included in this repository.

First, set up the environment and download the model weights from [Google Drive](https://drive.google.com/drive/folders/1pgYqyomDjzuUXPxTO543RYPyqej0wUen?usp=sharing). Then place the checkpoints in the expected folders.

<details>
<summary>Required checkpoint structure</summary>

```text
weights/
└── codeformer_swinir_tiny.pt

experiments/joint/tasktok_bl64/checkpoints/
├── tasktok_last.pt
├── token_predictor_last.pt
├── clsnet_last.pt
├── detnet_last.pt
└── segnet_last.pt
```

</details>

Then run the classification sanity check:

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch main/test_tasktok.py --config configs/tasktok_bl64_test.yaml --task cls
```

This command uses the small ImageNet-style sample pairs already included under:

```text
datasets/source/Imagenet/val-mixb/
├── gt/
└── lq/
```

The results will be saved to:

```text
experiments/joint/tasktok_bl64/test_results/
```

For a full evaluation on classification, segmentation, and detection, please prepare the full datasets as described in the Datasets section.

### 📦 Datasets

We use the [ImageNet](https://image-net.org/download-images) dataset for classification, and the [PASCAL VOC2012](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset/data) dataset for segmentation and detection.

Please place the datasets in the `datasets/source` folder following the required directory structure:

<details>
<summary>Dataset directory structure</summary>

```text
datasets/source/
├── Imagenet/
│   ├── train/
│   ├── val/
│   └── val-mixb/
│       ├── gt/
│       └── lq/
└── VOC/
    └── VOCdevkit/
        └── VOC2012/
            ├── Annotations/
            ├── ImageSets/
            ├── JPEGImages/
            ├── SegmentationClass/
            ├── JPEGImagesSeg-mixb/
            │   ├── gt/
            │   └── lq/
            └── JPEGImagesDet-mixb/
                ├── gt/
                └── lq/
```

</details>

To generate the degraded datasets, run:

```shell
# classification (ImageNet)
python datasets/val_data_generation/gen_cls-dataset.py --config datasets/val_data_generation/config/cls/imagenet-deg-mxb.yaml  # ImageNet for classification
# segmentation (VOC2012)
python datasets/val_data_generation/gen_seg-dataset.py --config datasets/val_data_generation/config/seg/pascalvoc-deg-mxb.yaml  # VOC2012 for segmentation
# detection (VOC2012)
python datasets/val_data_generation/gen_det-dataset.py --config datasets/val_data_generation/config/det/pascalvoc-deg-mxb.yaml  # VOC2012 for detection
```

By default, the generated degraded datasets are saved under the `experiments` directory. Please copy them to the expected dataset folders:

```shell
mkdir -p datasets/source/Imagenet/val-mixb
cp -r experiments/cls/imagenet/val-mixb/{gt,lq} datasets/source/Imagenet/val-mixb/

mkdir -p datasets/source/VOC/VOCdevkit/VOC2012/JPEGImagesSeg-mixb
cp -r experiments/seg/voc2012/pascalvoc-seg-mixb/{gt,lq} datasets/source/VOC/VOCdevkit/VOC2012/JPEGImagesSeg-mixb/

mkdir -p datasets/source/VOC/VOCdevkit/VOC2012/JPEGImagesDet-mixb
cp -r experiments/det/voc2012/pascalvoc-det-mixb/{gt,lq} datasets/source/VOC/VOCdevkit/VOC2012/JPEGImagesDet-mixb/
```

### ✅ Test

Model weights are available from [Google Drive](https://drive.google.com/drive/folders/1pgYqyomDjzuUXPxTO543RYPyqej0wUen?usp=sharing). Please download them and place them in the appropriate folders.

For testing, place the checkpoints as follows:

<details>
<summary>Checkpoint directory structure</summary>

```text
weights/
└── codeformer_swinir_tiny.pt

experiments/joint/tasktok_bl64/checkpoints/
├── tasktok_last.pt
├── token_predictor_last.pt
├── clsnet_last.pt
├── detnet_last.pt
└── segnet_last.pt

experiments/joint/tasktok_sl256/checkpoints/
├── tasktok_last.pt
├── token_predictor_last.pt
├── clsnet_last.pt
├── detnet_last.pt
└── segnet_last.pt
```

</details>

Then run:

```shell
# TaskTok-64
CUDA_VISIBLE_DEVICES=0 accelerate launch main/test_tasktok.py --config configs/tasktok_bl64_test.yaml

# TaskTok-256
CUDA_VISIBLE_DEVICES=0 accelerate launch main/test_tasktok.py --config configs/tasktok_sl256_test.yaml
```

### 🚀 Train

#### Greedy Search for Initialization (Optional)

You can skip this step if you use the precomputed greedy token orders provided in this repository.

```shell
CUDA_VISIBLE_DEVICES=0 python main/greedy_search.py --config configs/greedy_search_bl64.yaml --n_samples 300
```

Precomputed greedy token orders are also provided under:

```text
experiments/joint/greedy_search_bl64/greedy_token_order.pt
experiments/joint/greedy_search_sl256/greedy_token_order.pt
```

If you use the provided greedy token orders, please make sure the `greedy_token_order` path in the training config points to the corresponding file.

#### Start Training

Please download the files in the `weights` folder provided through Google Drive and place them in the following directory. The classification oracle model will be downloaded automatically and does not need to be placed manually.

<details>
<summary>Weight directory structure</summary>

```text
weights/
├── codeformer_swinir_tiny.pt
├── detnet_oracle.pt
└── segnet_oracle.pt
```

</details>

The SwinIR-Tiny training code and the segmentation/detection oracle models were trained using the [EDTR](https://github.com/JaehaKim97/EDTR) codebase.

```shell
# TaskTok-64
CUDA_VISIBLE_DEVICES=0 accelerate launch main/train_tasktok.py --config configs/tasktok_bl64.yaml

# TaskTok-256
CUDA_VISIBLE_DEVICES=0 accelerate launch main/train_tasktok.py --config configs/tasktok_sl256.yaml
```

### 🖼️ 512×512 Input Variant

We also provide an `input512` variant that directly supports 512×512 input images. This variant is based on TiTok (256x256), where only the input/output layers are modified for 512×512 resolution and the model is retrained accordingly.

Please check the [Google Drive](https://drive.google.com/drive/folders/1pgYqyomDjzuUXPxTO543RYPyqej0wUen?usp=sharing) `weights` folder for the modified TiTok checkpoints and the TaskTok checkpoints trained with them. To use this variant, run the corresponding `input512` config files:

```shell
# Test
CUDA_VISIBLE_DEVICES=0 accelerate launch main/test_tasktok.py --config configs/tasktok_bl64_input512_test.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch main/test_tasktok.py --config configs/tasktok_sl256_input512_test.yaml

# Train
CUDA_VISIBLE_DEVICES=0 accelerate launch main/train_tasktok.py --config configs/tasktok_bl64_input512.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch main/train_tasktok.py --config configs/tasktok_sl256_input512.yaml
```


### 📌 Citation

```bibtex
@inproceedings{lee2026tasktok,
  title={TaskTok: Delving into Task Tokens for Task-driven Image Restoration},
  author={Lee, Hongjae and Kang, Sojung and Yu, Jaeseong and Jung, Seung-Won},
  booktitle={European Conference on Computer Vision},
  year={2026}
}
```

### 📬 Contact

Email: [jimmy9704@korea.ac.kr](mailto:jimmy9704@korea.ac.kr)
