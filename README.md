[README.md](https://github.com/user-attachments/files/29925566/README.md)
# FBLSN
Generalizable Deepfake Detection with Frozen Backbone and Lightweight Side Network
# Frozen ViT with SAM and TMF for Generalizable Deepfake Detection

This repository contains the training and preprocessing code for a deepfake detection framework built on a frozen ViT-B/16 backbone. The model introduces a Self-Attention Modulation (SAM) module and a Token-based Multi-scale Feature Fusion (TMF) module to improve cross-dataset generalization.

## Repository Structure

```text
.
├── model_train.py       # Model definition, random video-level split, training, validation, and testing
├── preprocess.py        # FF++ c23 video preprocessing with MTCNN
├── checkpoint.pth       # Trained model checkpoint
├── requirements.txt     # Python dependencies
└── README.md
```

## Method Overview

The framework contains the following components:

- **Frozen ViT-B/16 backbone:** preserves the general visual representations learned from ImageNet-21K.
- **SAM:** generates spatial- and frequency-aware modulation terms for the Query, Key, and Value representations in all Transformer blocks.
- **TMF:** aligns and adaptively fuses features from three CNN stages at each spatial position.
- **Cross-attention injection:** injects fused multi-scale CNN priors into the 1st, 4th, and 7th ViT blocks.
- **Classification head:** predicts whether the input face image is real or fake.

Only the newly introduced modules and the classification head are optimized during training. The ViT backbone remains frozen.

## Requirements

Recommended environment:

- Python 3.10
- CUDA-compatible NVIDIA GPU
- PyTorch with a CUDA version compatible with the local system

Install the required packages:

```bash
pip install -r requirements.txt
```

> PyTorch installation depends on the CUDA version of the local machine. When necessary, install PyTorch from the official PyTorch installation page first, and then install the remaining dependencies.

## Datasets

The experiments use four public deepfake datasets.

### FaceForensics++ (FF++)

Official repository and access instructions:

- https://github.com/ondyari/FaceForensics
- https://github.com/ondyari/FaceForensics/tree/master/dataset

This project uses the **c23** compression setting. The following manipulation methods are supported:

- Deepfakes
- Face2Face
- FaceSwap
- FaceShifter
- NeuralTextures

Expected raw FF++ structure:

```text
FF++/
├── manipulated_sequences/
│   ├── Deepfakes/
│   │   ├── c23/
│   │   │   └── videos/
│   │   │       ├── 000_003.mp4
│   │   │       ├── 001_870.mp4
│   │   │       └── ...
│   │   └── c40/
│   │       └── videos/
│   ├── Face2Face/
│   │   ├── c23/
│   │   │   └── videos/
│   │   │       ├── 000_003.mp4
│   │   │       ├── 001_870.mp4
│   │   │       ├── 002_006.mp4
│   │   │       └── ...
│   │   ├── c40/
│   │   │   └── videos/
│   │   └── masks/
│   ├── FaceShifter/
│   │   ├── c23/
│   │   │   └── videos/
│   │   └── c40/
│   │       └── videos/
│   ├── FaceSwap/
│   │   ├── c23/
│   │   │   └── videos/
│   │   └── c40/
│   │       └── videos/
│   └── NeuralTextures/
│       ├── c23/
│       │   └── videos/
│       └── c40/
│           └── videos/
└── original_sequences/
    └── youtube/
        ├── c23/
        │   └── videos/
        │       ├── 000.mp4
        │       ├── 001.mp4
        │       └── ...
        └── c40/
            └── videos/
```

For the provided preprocessing script, set:

```python
INPUT_ROOT = "FF++"
```

The script reads only the `c23/videos` directories during preprocessing.

### Celeb-DF-v2 (CDF)

Official project pages:

- https://github.com/yuezunli/celeb-deepfakeforensics
- https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics.html

### WildDeepfake (WDF)

Official repository and download instructions:

- https://github.com/OpenTAI/wild-deepfake

### Deepfake Detection Challenge (DFDC)

Official dataset pages:

- https://ai.meta.com/datasets/dfdc/
- https://www.kaggle.com/competitions/deepfake-detection-challenge/data

> The datasets are not redistributed in this repository. Please follow the licenses, access rules, and terms specified by the respective dataset providers.

## FF++ Preprocessing

The preprocessing script performs the following operations:

1. Reads FF++ c23 videos.
2. Uniformly samples 32 frames from each video.
3. Detects faces using MTCNN.
4. Selects and crops the largest detected face.
5. Expands the face region by a scale factor of 1.3.
6. Resizes each cropped face to `224 × 224`.
7. Saves the processed frames as JPEG images.

Before running the script, edit the following paths in `preprocess.py`:

```python
INPUT_ROOT = "ALLff++"
OUTPUT_ROOT = "preprocessed_FFPP_mtcnn"
```

Run preprocessing:

```bash
python preprocess.py
```

Expected output structure:

```text
preprocessed_FFPP_mtcnn/
├── original/
│   ├── 000/
│   │   ├── img_0001.jpg
│   │   └── ...
│   └── ...
├── Deepfakes/
├── Face2Face/
├── FaceShifter/
├── FaceSwap/
└── NeuralTextures/
```

## Training

Before training, edit the dataset root and checkpoint path in `model_train.py`:

```python
ROOT = r"path\to\preprocessed_FFPP_mtcnn"
SAVE_PATH = "checkpoint.pth"
```

Run training:

```bash
python model_train.py
```

Default training settings:

| Setting | Value |
|---|---:|
| Backbone | ViT-B/16 pretrained on ImageNet-21K |
| Input resolution | 224 × 224 |
| Batch size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| Initial learning rate | 1e-4 |
| Frames per video | 32 |
| Pooling | Mean pooling |
| Random seed | 42 |

## Video-Level Data Split

The training script performs a random split at the **source-video level** rather than loading a predefined split file.

The default numbers of real source videos are:

```python
N_TRAIN_REAL_VID = 700
N_VAL_REAL_VID = 100
N_TEST_REAL_VID = 200
```

To prevent data leakage:

- all frames from the same video are assigned to only one subset;
- each real source video and all forged videos generated from that source are kept in the same subset;
- source-video IDs are checked to ensure that the training, validation, and test sets do not overlap.

The split is reproducible when the same random seed is used. Change `SEED` to generate another random video-level split.

## Evaluation

During training, the model is evaluated on the validation set after each epoch. The checkpoint with the highest validation AUC is saved to:

```text
checkpoint.pth
```

After training, the best checkpoint is evaluated on the test set.

Reported metrics include:

- Accuracy
- AUC
- Precision
- Recall
- F1-score
- Confusion matrix

## Loading the Trained Checkpoint

A checkpoint can be loaded as follows:

```python
import torch

model = ModelFull(
    vit_name=CFG.VIT_NAME,
    img_size=CFG.IMAGE_SIZE,
    patch_size=CFG.PATCH_SIZE,
    pool=CFG.POOL,
    num_classes=2,
)

state_dict = torch.load("checkpoint.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()
```

## Checkpoint Upload

GitHub blocks ordinary files larger than 100 MB. If `checkpoint.pth` exceeds this limit, use one of the following approaches:

### Git LFS

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes checkpoint.pth
git commit -m "Add trained checkpoint"
git push
```

### GitHub Release

Alternatively, upload `checkpoint.pth` as a release asset and place its download link in this README.

## Notes

- The preprocessing script uses the `mtcnn` Python package and OpenCV.
- The training script uses the Hugging Face `transformers` implementation of ViT.
- The pretrained ViT weights are downloaded automatically when `AutoModel.from_pretrained(...)` is first called.
- The frozen ViT backbone still participates in the forward pass and therefore contributes to the computational cost.
- Dataset paths and the number of DataLoader workers may need to be adjusted for the local environment.

## Citation

Please add the BibTeX entry of the corresponding paper after publication:

```bibtex
@article{fu2026generalizable,
  title   = {Your Paper Title},
  author  = {Fu, Jiahao and Meng, Yongwei and Yang, Tao and Zhou, Cheng and Hu, Jiale},
  journal = {The Visual Computer},
  year    = {2026}
}
```

## License

This repository is intended for academic research. The source code license does not cover the four datasets. Each dataset remains subject to the license and terms of its original provider.
