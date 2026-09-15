# Ternary ResNet on CIFAR-10 with Knowledge Distillation

CIFAR-10 image classification using ResNet knowledge distillation and ternary-weight quantization-aware training (QAT).

This project trains a full-precision ResNet-34 teacher, a full-precision ResNet-18 baseline, and a ternary ResNet-18 student using knowledge distillation and quantization-aware training.

## Dataset

The experiments use the **CIFAR-10 dataset** provided as a `cifar.zip` archive in the Kaggle environment.

The dataset is loaded directly from this archive. **No additional CIFAR-10 download is required.**

The data is split into:

* Training: 45,000 images
* Validation: 5,000 images
* Official test set: 10,000 images

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

Large experiment-output archives are provided through **GitHub Releases** rather than stored directly in the repository.

## Notebooks

### 01 — Teacher ResNet-34

Trains a full-precision ResNet-34 from scratch on CIFAR-10.

The trained teacher is later frozen and used to provide soft predictions for knowledge distillation.

### 02 — FP32 ResNet-18 Baseline

Trains a standard full-precision ResNet-18 from scratch.

This model uses:

* No knowledge distillation
* No ternary quantization
* Standard FP32 weights

It provides the primary baseline for evaluating the ternary student.

### 03 — Ternary KD/QAT Student

Trains a ResNet-18 student using:

* Knowledge distillation from the trained ResNet-34 teacher
* Ternary-weight quantization-aware training
* Straight-Through Estimator (STE)
* Latent FP32 weights during optimization

The internal convolutional weights are ternarized during the forward pass to:

```text
{-α, 0, +α}
```

The first convolutional layer (`conv1`), final fully connected layer (`fc`), and BatchNorm parameters remain in FP32.

Latent FP32 weights are maintained during optimization, while ternary weights are used during the forward computation of the quantized convolutional layers.

The primary knowledge-distillation configuration uses:

* Temperature: `T = 4`
* KD loss weight: `λ = 0.7`

The training objective is:

```text
L = (1 - λ) × L_CE + λ × T² × L_KD
```

where:

* `L_CE` is the cross-entropy loss using the ground-truth labels
* `L_KD` is the knowledge-distillation loss between teacher and student soft predictions
* `T` is the distillation temperature
* `λ` controls the contribution of the distillation loss

### 04 — Evaluation and Ablation

Performs final evaluation and analysis of the trained models.

The notebook compares:

* Test accuracy
* Parameter count
* Model storage size
* Ternary sparsity
* Compression ratio
* Weight distributions
* Knowledge-distillation temperature ablation
* Compact checkpoint reconstruction and reload verification

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

This provides the complete workflow from dataset preparation and model training to final evaluation, compression analysis, and ablation experiments.

## Training Setup

Common experiment settings include:

* Batch size: 128
* Random seed: 42
* Random crop with padding: 4
* Random horizontal flip
* Cutout: 8
* CIFAR-10 normalization
* SGD optimizer with momentum
* Learning-rate warmup
* Learning-rate scheduling
* Early stopping
* Maximum training epochs: 200

The exact configuration for each experiment is implemented in its corresponding notebook.

## Final Results

| Model                    | Test Accuracy | Parameters |      Model Storage |
| ------------------------ | ------------: | ---------: | -----------------: |
| ResNet-34 Teacher        |        95.40% | 21,282,122 |   81.3128 MiB FP32 |
| ResNet-18 FP32 Baseline  |        95.49% | 11,173,962 |   42.6984 MiB FP32 |
| ResNet-18 Ternary KD/QAT |        95.32% | 11,173,962 | 2.8212 MiB compact |

The final ternary ResNet-18 student achieves **95.32% test accuracy**.

Compared with the FP32 ResNet-18 baseline:

* FP32 baseline storage: **42.6984 MiB**
* Compact ternary storage: **2.8212 MiB**
* Storage compression: approximately **15.135×**

The ternary convolutional weights have approximately **45.5% sparsity**.

Compared with the ResNet-34 teacher:

* Teacher accuracy: **95.40%**
* Ternary student accuracy: **95.32%**
* Accuracy difference: **0.08 percentage points**

The student therefore retains nearly all of the teacher's classification accuracy while substantially reducing model-storage requirements.

## Knowledge-Distillation Ablation

A temperature ablation is performed to study the effect of the knowledge-distillation temperature while keeping the KD loss weight fixed.

| Temperature |   λ | Test Accuracy |
| ----------- | --: | ------------: |
| 4           | 0.7 |        95.32% |
| 2           | 0.7 |        95.17% |

The main experiment uses `T = 4` and `λ = 0.7`.

In these experiments, `T = 4` produced slightly better final test accuracy than `T = 2`.

## Ternary Quantization

For the quantized convolutional layers, FP32 weights are mapped to three possible values during the forward pass:

```text
{-α, 0, +α}
```

A threshold determines whether each weight is mapped to the negative ternary value, zero, or the positive ternary value.

The scaling factor `α` represents the magnitude used for the non-zero ternary weights.

The Straight-Through Estimator allows gradients to propagate through the non-differentiable ternarization operation during backpropagation.

This enables:

* Ternary forward-pass weights
* FP32 latent weights for optimization
* End-to-end training using standard gradient-based optimization

## Compact Model Storage

The reported **2.8212 MiB** student model size refers to the exported **compact ternary representation**.

It is not the size of the latent-FP32 QAT training checkpoint.

During training, FP32 latent weights are maintained so that gradient-based optimization can update the network.

For compact storage, the ternary model representation stores the ternary weight information more efficiently instead of storing every quantized convolutional weight as a standard 32-bit floating-point value.

The evaluation notebook verifies that the compact representation can be reconstructed and reloaded correctly for inference.

## Output Files

The experiment outputs are provided as ZIP archives:

* `Teacher_ResNet34_Outputs.zip`
* `Baseline_ResNet18_Outputs.zip`
* `Ternary_KD_QAT_Student_Outputs.zip`
* `Evaluation_Ablation_Outputs.zip`

These archives contain the corresponding:

* Trained checkpoints
* Training logs
* Accuracy and loss curves
* Evaluation results
* Plots
* Compression artifacts
* Ablation outputs

Because these files are large, they are distributed through **GitHub Releases** instead of being committed directly to the repository.

## Requirements

The project is designed to run in a Kaggle GPU environment.

Main dependencies include:

* Python 3.x
* PyTorch
* torchvision
* NumPy
* Pandas
* Matplotlib
* Pillow
* tqdm

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducibility

To reproduce the experiments:

1. Add the CIFAR-10 archive to the Kaggle environment as `cifar.zip`.
2. Run the notebooks in the specified order.
3. Use the provided training configuration and random seed.
4. Use the best saved checkpoint from each stage for the following experiments.

The experiments use a fixed random seed of:

```text
42
```

The official CIFAR-10 test set is reserved exclusively for final evaluation.

## Important Implementation Details

* The ResNet-34 teacher is trained in full precision.
* The ResNet-18 baseline is trained independently in full precision.
* The teacher is frozen during student knowledge-distillation training.
* The student uses ternary internal convolutional weights during the forward pass.
* The first convolutional layer, final fully connected layer, and BatchNorm parameters remain in FP32.
* Latent FP32 weights are maintained during QAT for optimization.
* The Straight-Through Estimator is used to propagate gradients through ternary quantization.
* The validation set is used for model selection and early stopping.
* The official CIFAR-10 test set is not used during training.
* The reported compact model size refers to the exported ternary representation rather than the latent-FP32 training checkpoint.

## Notes

The reported compression represents a reduction in **model-storage requirements**.

It should not be interpreted as an equivalent inference-speed improvement.

Actual inference acceleration depends on hardware and software support for efficient ternary arithmetic. Standard GPU and CPU libraries may internally convert or process ternary values using conventional numerical representations, so storage compression does not automatically translate into proportional latency or throughput improvements.

## Summary

The project demonstrates that a ResNet-18 student can combine knowledge distillation with ternary-weight quantization-aware training while retaining high CIFAR-10 classification accuracy.

The final student achieves:

```text
Test Accuracy:        95.32%
Parameters:           11,173,962
Compact Model Size:   2.8212 MiB
Ternary Sparsity:     ~45.5%
Storage Compression:  ~15.135×
```

Compared with the ResNet-34 teacher, the ternary student loses only **0.08 percentage points** of test accuracy while providing a substantially more compact model representation.
