[README(2).md](https://github.com/user-attachments/files/29925694/README.2.md)
# Deepfake Detection

## Datasets

The following four public datasets are used in this project:

- **FaceForensics++ (FF++)**  
  https://github.com/ondyari/FaceForensics

- **Celeb-DF-v2 (CDF)**  
  https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics.html

- **WildDeepfake (WDF)**  
  https://github.com/OpenTAI/wild-deepfake

- **Deepfake Detection Challenge (DFDC)**  
  https://ai.meta.com/datasets/dfdc/

Please follow the access requirements and licenses specified by the respective dataset providers.

## Environment Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The project files are organized as follows:

```text
.
├── model_train.py
├── preprocess.py
├── checkpoint.pth
├── requirements.txt
└── README.md
```

## Download FaceForensics++

Download the FaceForensics++ dataset using the official download instructions and rename its root directory to:

```text
ALLff++/
```

The raw dataset should be organized as follows:

```text
ALLff++/
├── manipulated_sequences/
│   ├── Deepfakes/
│   │   └── c23/
│   │       └── videos/
│   ├── Face2Face/
│   │   └── c23/
│   │       └── videos/
│   │           ├── 000_003.mp4
│   │           ├── 001_870.mp4
│   │           └── ...
│   ├── FaceShifter/
│   │   └── c23/
│   │       └── videos/
│   ├── FaceSwap/
│   │   └── c23/
│   │       └── videos/
│   └── NeuralTextures/
│       └── c23/
│           └── videos/
└── original_sequences/
    └── youtube/
        └── c23/
            └── videos/
                ├── 000.mp4
                ├── 001.mp4
                └── ...
```

## Preprocess FaceForensics++

Set the input and output directories in `preprocess.py` as follows:

```python
INPUT_ROOT = "ALLff++"
OUTPUT_ROOT = "FF++"
```

Run the preprocessing script:

```bash
python preprocess.py
```

The script uniformly samples 32 frames from each video, detects and crops faces using MTCNN, resizes them to `224 × 224`, and saves the resulting images.

After preprocessing, the output directory should be organized as follows:

```text
FF++/
├── Deepfakes/
│   ├── 000_003/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── ...
├── Face2Face/
│   ├── 000_003/
│   ├── 001_870/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   ├── img_0003.jpg
│   │   └── ...
│   └── ...
├── FaceShifter/
├── FaceSwap/
├── NeuralTextures/
└── original/
    ├── 000/
    │   ├── img_0001.jpg
    │   ├── img_0002.jpg
    │   └── ...
    └── ...
```

Set the dataset path in `model_train.py` to the preprocessed directory:

```python
ROOT = r"path\to\FF++"
```
