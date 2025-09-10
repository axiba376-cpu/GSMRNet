# GSMRNet

A deep learning approach for bridge defects detection using GSMRNet.

## Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
PIL>=8.3.0
numpy>=1.21.0
einops>=0.3.0
matplotlib>=3.4.0
```

## Dataset

Download the open-source dataset from:
- **Link**: https://pan.quark.cn/s/42d06ae9f45b
- **Extract Code**: 2Hu8

Organize the dataset as follows:
```
datasets/
├── crack500_train/
│   ├── train/
│   │   ├── a/          # Input images
│   │   └── b/          # Target images
│   └── validation/
│       ├── a/          # Validation input images
│       └── b/          # Validation target images
```

## Training

Run training with default parameters:
```bash
python train.py
```

Or customize training parameters:
```bash
python train.py --dataset crack500_train \
                --batch_size 4 \
                --num_epochs 300 \
                --lrG 0.0001 \
                --lrD 0.0004
```

## Testing

For testing, organize test data:
```
dataset/
└── crack500_test/
    └── test_folder_all/
        ├── a/          # Test input images
        ├── b/          # Test target images
        └── label/      # Ground truth labels
```

Run testing:
```bash
python test.py
```

Or with custom parameters:
```bash
python test.py --dataset crack500_test \
               --direction BtoA \
               --batch_size 1
```

## Results

- Trained models are saved in `./saved-model_crack500/`
- Test results are saved in `./save/` and `./result-crack/`
