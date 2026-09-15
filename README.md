# Ternary ResNet on CIFAR-10 with Knowledge Distillation

CIFAR-10 image classification using ResNet knowledge distillation and ternary-weight quantization-aware training (QAT).

This project trains a full-precision ResNet-34 teacher, a full-precision ResNet-18 baseline, and a ternary ResNet-18 student using knowledge distillation and quantization-aware training.

## Dataset

The experiments use the **CIFAR-10 dataset** provided as a `cifar.zip` archive in the Kaggle environment.

The dataset is loaded directly from this archive. **No additional CIFAR-10 download is required.**

The data is split into:

- Training: 45,000 images
- Validation: 5,000 images
- Official test set: 10,000 images

The official CIFAR-10 test set is kept untouched during training and model selection and is used only for final evaluation.

## Project Structure

```text
Ternary-resnet-cifar10/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_Teacher_ResNet34.ipynb
│   ├── 02_Baseline_ResNet18.ipynb
│   ├── 03_Ternary_KD_QAT_Student.ipynb
│   └── 04_Evaluation_Ablation_Final_Analysis.ipynb
│
└── outputs/
    └── README.md
```

## Notebooks

### 01 — Teacher ResNet-34

Trains a full-precision ResNet-34 from scratch on CIFAR-10.

The trained teacher is later frozen and used to provide soft predictions for knowledge distillation.

### 02 — FP32 ResNet-18 Baseline

Trains a standard full-precision ResNet-18 from scratch without knowledge distillation or quantization.

This provides the baseline for comparison with the ternary student.

### 03 — Ternary KD/QAT Student

Trains a ResNet-18 student using:

- Knowledge distillation from the trained ResNet-34 teacher
- Ternary-weight quantization-aware training
- Straight-Through Estimator (STE)
- Latent FP32 weights during optimization

The internal convolutional weights are ternarized during the forward pass to:

```text
{-α, 0, +α}
```

The first convolutional layer (`conv1`), final fully connected layer (`fc`), and BatchNorm parameters remain in FP32.

The primary knowledge-distillation configuration uses:

- Temperature: `T = 4`
- KD loss weight: `λ = 0.7`

### 04 — Evaluation and Ablation

Evaluates the trained models and compares:

- Test accuracy
- Parameter count
- Model storage size
- Ternary sparsity
- Compression
- Weight distributions
- Knowledge-distillation temperature ablation
- Compact checkpoint reload verification

## Training Pipeline

Run the notebooks in the following order:

```text
01_Teacher_ResNet34.ipynb
        ↓
02_Baseline_ResNet18.ipynb
        ↓
03_Ternary_KD_QAT_Student.ipynb
        ↓
04_Evaluation_Ablation_Final_Analysis.ipynb
```

This forms the complete workflow from dataset loading and model training to final evaluation and ablation analysis.

## Training Setup

Common training settings include:

- Batch size: 128
- Random seed: 42
- Random crop with padding: 4
- Random horizontal flip
- Cutout: 8
- CIFAR-10 normalization
- SGD with momentum
- Learning-rate warmup
- Learning-rate scheduling
- Early stopping
- Maximum training epochs: 200

The exact configurations are implemented in the corresponding notebooks.

## Final Results

| Model | Test Accuracy | Parameters | Model Storage |
|---|---:|---:|---:|
| ResNet-34 Teacher | 95.40% | 21,282,122 | 81.3128 MiB |
| ResNet-18 FP32 Baseline | 95.49% | 11,173,962 | 42.6984 MiB |
| ResNet-18 Ternary KD/QAT | 95.32% | 11,173,962 | 2.8212 MiB |

The final ternary ResNet-18 student achieves **95.32% test accuracy** while reducing the FP32 ResNet-18 model storage by approximately **15.135×**.

The ternary convolutional weights have approximately **45.5% sparsity**.

The student is only **0.08 percentage points below the ResNet-34 teacher**.

## Outputs

The notebooks generate the following experiment artifacts:

- Trained model checkpoints
- Training and validation logs
- Accuracy and loss curves
- Learning-rate curves
- Evaluation results
- Weight-distribution visualizations
- Compression and sparsity analysis
- Knowledge-distillation ablation results

The final trained checkpoints and generated plots are retained with the experiment outputs.

## Requirements

The project is designed to run in a Kaggle GPU environment.

Main dependencies:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- Pillow
- tqdm

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducibility

Run the notebooks in the order listed above.

The experiments use a fixed random seed of:

```text
42
```

The CIFAR-10 dataset should be provided through the `cifar.zip` archive used in the Kaggle environment.

The official CIFAR-10 test set is kept separate from training and validation and is used only for final evaluation.
