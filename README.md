# Ternary ResNet on CIFAR-10 with Knowledge Distillation

CIFAR-10 image classification using ResNet knowledge distillation and ternary-weight quantization-aware training (QAT).

This project investigates whether a ResNet-18 model can maintain high classification accuracy while significantly reducing model storage through ternary-weight quantization. A full-precision ResNet-34 is first trained as a teacher model, followed by a full-precision ResNet-18 baseline. The final student model combines knowledge distillation with ternary-weight quantization-aware training.

---

## Project Overview

The project consists of four main stages:

1. **ResNet-34 Teacher**
   - Full-precision ResNet-34 trained from scratch.
   - Provides the teacher predictions used for knowledge distillation.

2. **FP32 ResNet-18 Baseline**
   - Standard full-precision ResNet-18 trained from scratch.
   - Provides a baseline for evaluating the effect of ternary quantization and knowledge distillation.

3. **Ternary ResNet-18 Student**
   - ResNet-18 trained using knowledge distillation and ternary-weight QAT.
   - Internal convolutional weights are quantized to: `{-α, 0, +α}`.
   - Latent FP32 weights are maintained during training.
   - Straight-Through Estimation (STE) is used to propagate gradients through the quantization operation.

4. **Evaluation and Ablation**
   - Compares the teacher, FP32 baseline, and ternary student.
   - Evaluates accuracy, parameter count, sparsity, checkpoint size, and compression.
   - Includes ternary-weight verification, weight-distribution visualization, and a knowledge-distillation temperature ablation.

---

## Dataset

The experiments use the **CIFAR-10 dataset**.

The CIFAR-10 dataset was provided as a `cifar.zip` archive and used as the dataset source in the Kaggle environment.

The ZIP archive contains the CIFAR-10 Python dataset files used by the notebooks, including the training batches and official test batch.

**No additional CIFAR-10 download is required.**

The dataset is divided into:

- **Training set:** 45,000 images
- **Validation set:** 5,000 images
- **Official CIFAR-10 test set:** 10,000 images

The official test set is kept untouched during training and model selection and is used only for final evaluation.

---

## Repository Structure

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

The trained models, logs, plots, checkpoints, and other experiment artifacts are provided separately as ZIP archives through GitHub Releases because the output archives are larger than GitHub's normal browser upload limit.

---

## Experimental Pipeline

The complete workflow is:

```text
CIFAR-10 Dataset
       │
       ▼
Data Preprocessing & Augmentation
       │
       ├───────────────────────┐
       ▼                       ▼
ResNet-34 Teacher       FP32 ResNet-18 Baseline
       │                       │
       │                       │
       └───────────┬───────────┘
                    ▼
       Ternary ResNet-18 Student
       + Knowledge Distillation
       + Quantization-Aware Training
                    │
                    ▼
        Evaluation & Ablation
                    │
                    ▼
       Accuracy / Compression /
       Sparsity / Weight Analysis
```

The notebooks should be executed in the following order:

```text
01_Teacher_ResNet34.ipynb
        ↓
02_Baseline_ResNet18.ipynb
        ↓
03_Ternary_KD_QAT_Student.ipynb
        ↓
04_Evaluation_Ablation_Final_Analysis.ipynb
```

This provides an end-to-end workflow from dataset loading and teacher training through student training, evaluation, compression analysis, and ablation experiments.

---

## Data Preprocessing and Augmentation

The following preprocessing and augmentation techniques are used during training:

- Random crop with padding of 4 pixels
- Random horizontal flip
- Cutout with size 8
- CIFAR-10 normalization

The normalization values are:

- Mean = (0.4914, 0.4822, 0.4465)
- Std = (0.2470, 0.2435, 0.2616)

**Batch size:** 128

**Random seed:** 42

The same general CIFAR-10 preprocessing pipeline is used across the teacher, baseline, and student experiments.

---

## Model Architectures

### ResNet-34 Teacher

A full-precision ResNet-34 is trained from scratch and used as the teacher network.

The teacher produces the soft prediction distribution used during knowledge-distillation training of the student.

### FP32 ResNet-18 Baseline

A standard full-precision ResNet-18 is trained independently.

The baseline does not use:

- Knowledge distillation
- Ternary quantization
- Quantization-aware training

Its purpose is to provide a reference point for measuring the effect of the proposed ternary KD/QAT approach.

### Ternary ResNet-18 Student

The final student uses the ResNet-18 architecture.

The student contains:

- Ternary internal convolutional weights
- FP32 input convolution (conv1)
- FP32 fully connected layer (fc)
- FP32 batch-normalization parameters
- Latent FP32 weights during optimization
- Straight-Through Estimator for gradient propagation

The ternary convolutional weights used during the forward pass take the form:

```text
{-α, 0, +α}
```

---

## Training Configuration

### ResNet-34 Teacher

The teacher model is trained using:

| Parameter | Setting |
|---|---|
| Architecture | ResNet-34 |
| Precision | FP32 |
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Momentum | 0.9 |
| Nesterov | Enabled |
| Weight Decay | 5e-4 |
| Label Smoothing | 0.1 |
| Warmup | 5 epochs |
| LR Scheduler | ReduceLROnPlateau |
| Scheduler Factor | 0.5 |
| Scheduler Patience | 8 |
| Minimum LR | 1e-4 |
| Maximum Epochs | 200 |
| Early Stopping Patience | 20 |
| Best Model Criterion | Validation Accuracy |

A 5-epoch linear learning-rate warmup is used before the main learning-rate schedule.

### FP32 ResNet-18 Baseline

The baseline ResNet-18 uses full-precision weights and does not use knowledge distillation or quantization.

The baseline follows the same general CIFAR-10 training pipeline as the teacher, providing a fair comparison with the ternary student.

| Parameter | Setting |
|---|---|
| Architecture | ResNet-18 |
| Precision | FP32 |
| Optimizer | SGD |
| Momentum | 0.9 |
| Nesterov | Enabled |
| Weight Decay | 5e-4 |
| Batch Size | 128 |
| Maximum Epochs | 200 |
| Early Stopping | Enabled |
| Knowledge Distillation | No |
| Quantization | No |

### Ternary KD/QAT Student

The final student uses the following configuration:

| Parameter | Setting |
|---|---|
| Architecture | ResNet-18 |
| Initialization | Trained FP32 ResNet-18 |
| Quantization | Ternary |
| Forward Weights | Ternary |
| Latent Weights | FP32 |
| STE | Enabled |
| KD Temperature | 4 |
| KD Loss Weight | 0.7 |
| Threshold Factor | 0.7 |
| Optimizer | SGD |
| Learning Rate | 0.002 |
| Momentum | 0.9 |
| Nesterov | Enabled |
| Weight Decay | 1e-4 |
| Gradient Clipping | Maximum norm 5 |
| Warmup | 5 epochs |
| LR Scheduler | ReduceLROnPlateau |
| Scheduler Factor | 0.5 |
| Scheduler Patience | 8 |
| Minimum LR | 1e-5 |
| Maximum Epochs | 200 |
| Early Stopping Patience | 25 |

---

## Ternary Weight Quantization

The student uses ternary quantization for its internal convolutional weights.

For each output channel, a threshold is calculated from the latent FP32 weights:

```text
Δc = δ × mean(|Wc|)
```

where:

```text
δ = 0.7
```

The scaling factor is calculated using the weights whose magnitude exceeds the threshold:

```text
        sum(|Wci| × I(|Wci| > Δc))
αc = --------------------------------
        sum(I(|Wci| > Δc))
```

The quantized weight is then defined as:

```text
              +αc     if Wci > Δc

Wq,ci =        0      if |Wci| ≤ Δc

              -αc     if Wci < -Δc
```

Therefore, the quantized weights take only three possible values:

```text
-αc, 0, +αc
```

The scaling factor is calculated per output channel.

---

## Straight-Through Estimator

The ternary quantization operation is non-differentiable.

To allow gradient-based optimization, a Straight-Through Estimator (STE) is used.

During the forward pass, the ternary weights are used:

```text
Wq
```

while gradients are propagated to the underlying latent FP32 weights.

The implementation uses the equivalent formulation:

```text
WSTE = W + (Wq - W).detach()
```

This allows the forward pass to use ternary weights while the FP32 latent weights continue to receive gradients and be updated by the optimizer.

---

## Knowledge Distillation

Knowledge distillation transfers information from the trained ResNet-34 teacher to the ResNet-18 student.

The teacher is frozen during student training.

The student objective combines:

- Standard cross-entropy loss using the ground-truth labels.
- Temperature-scaled KL-divergence between teacher and student predictions.

The overall loss is:

```text
L = (1 - λ) LCE
    + λ T² DKL(
        softmax(zt / T)
        ||
        softmax(zs / T)
      )
```

where:

- zt = teacher logits
- zs = student logits
- T = distillation temperature
- λ = KD loss weight

The main experiment uses:

- T = 4
- λ = 0.7

---

## Training Results

The final test accuracies obtained from the experiments are:

| Model | Test Accuracy |
|---|---|
| ResNet-34 Teacher | 95.40% |
| FP32 ResNet-18 Baseline | 95.49% |
| Ternary ResNet-18 Student | 95.32% |

The ternary student therefore achieves a test accuracy very close to both the teacher and the full-precision ResNet-18 baseline.

The accuracy difference between the teacher and ternary student is:

```text
95.40% - 95.32% = 0.08 percentage points
```

---

## Training Details

### Teacher

The teacher achieved:

- Best Validation Accuracy: 96.12%
- Best Epoch: 129
- Final Test Accuracy: 95.40%
- Training Ended: Epoch 149

### Ternary Student

The main ternary KD/QAT student achieved:

- Best Validation Accuracy: 95.60%
- Best Epoch: 93
- Final Test Accuracy: 95.32%
- Training Ended: Epoch 118

The best checkpoint is selected based on validation accuracy.

---

## Compression and Model Size

The final models were evaluated in terms of parameter count, checkpoint storage, sparsity, and compression.

| Model | Test Accuracy | Parameters | Representation | Sparsity | Actual Checkpoint |
|---|---|---|---|---|---|
| ResNet-34 Teacher | 95.40% | 21,282,122 | FP32 | 0% | 81.3128 MiB |
| ResNet-18 Baseline | 95.49% | 11,173,962 | FP32 | 0% | 42.6984 MiB |
| Ternary ResNet-18 Student | 95.32% | 11,173,962 | Ternary + FP32 components | ~45.5% | 2.8212 MiB |

The theoretical ternary storage requirement for the student is approximately:

```text
2.7778 MiB
```

The actual compact checkpoint size is:

```text
2.8212 MiB
```

The resulting actual compression relative to the FP32 ResNet-18 checkpoint is approximately:

```text
15.135×
```

This demonstrates a substantial reduction in model storage while retaining high classification accuracy.

---

## Ternary Weight Sparsity

The ternary student contains a significant number of zero-valued weights.

The evaluation measured approximately:

```text
Zero weights: 5,079,809
```

The resulting sparsity of the ternary convolutional/fully connected weights is approximately:

```text
45.5%
```

This means that a substantial portion of the quantized weights are represented by zero, while the remaining non-zero weights take positive or negative scaled values.

---

## Knowledge Distillation Ablation

An ablation experiment was performed to study the effect of the distillation temperature.

The main experiment uses:

- T = 4
- λ = 0.7

A second experiment uses:

- T = 2
- λ = 0.7

The results are:

| Configuration | Validation Accuracy | Test Accuracy | Sparsity |
|---|---|---|---|
| T=4, λ=0.7 | 95.60% | 95.32% | 45.53% |
| T=2, λ=0.7 | 95.78% | 95.17% | 45.78% |

The main T=4 configuration achieved the higher test accuracy.

Training curves and additional analysis for this ablation are included in the evaluation notebook.

---

## Ternary Weight Verification

The final student checkpoint was also verified after saving and reloading the compact ternary representation.

The reloaded checkpoint reproduced the same test accuracy:

- Original Student Accuracy: 95.32%
- Reloaded Student Accuracy: 95.32%
- Difference: 0.00 percentage points

This confirms that the compact ternary checkpoint can be restored and evaluated without changing the final prediction accuracy.

---

## Weight Distribution

The evaluation includes a comparison of the FP32 and ternary weight distributions.

The FP32 model contains a continuous distribution of weight values, whereas the ternary student concentrates the quantized weights around three values for each channel:

```text
-α
 0
+α
```

This provides a direct visual verification that the intended ternary quantization is being applied.

---

## Accuracy and Compression Trade-off

The results demonstrate a clear trade-off between model precision and storage requirements.

The FP32 ResNet-18 achieves:

```text
95.49% test accuracy
```

while the ternary KD/QAT student achieves:

```text
95.32% test accuracy
```

The accuracy difference is only:

```text
0.17 percentage points
```

while the student checkpoint is approximately:

```text
15.135× smaller
```

than the FP32 ResNet-18 checkpoint.

The student is also only:

```text
0.08 percentage points
```

below the ResNet-34 teacher.

These results indicate that knowledge distillation combined with ternary QAT can retain high classification performance while substantially reducing model storage.

---

## Important Limitations

Although the ternary student provides significant storage compression, storage compression should not automatically be interpreted as the same reduction in inference latency or computational cost.

Efficient inference would require suitable ternary-weight kernels, optimized implementations, or hardware support.

In this implementation:

- conv1 remains FP32.
- fc remains FP32.
- Batch-normalization parameters remain FP32.
- Per-channel scaling factors introduce additional storage overhead.
- The reported compression is based on the compact checkpoint representation.
- The experiments are performed only on CIFAR-10.

Therefore, the results demonstrate the effectiveness of ternary-weight representation for this experimental setting, but they do not directly establish an equivalent 15.135× inference speedup.

---

## Reproducibility

All experiments use a fixed random seed:

```text
42
```

The recommended execution order is:

1. `01_Teacher_ResNet34.ipynb`
2. `02_Baseline_ResNet18.ipynb`
3. `03_Ternary_KD_QAT_Student.ipynb`
4. `04_Evaluation_Ablation_Final_Analysis.ipynb`

The notebooks contain the complete training and evaluation procedures.

The experiments are intended to run in a Kaggle GPU environment using the provided CIFAR-10 dataset archive.

---

## Output Files

The experiment outputs are provided as separate ZIP archives.

The output archives contain the trained checkpoints, logs, plots, and other generated artifacts from the corresponding notebooks.

The four output packages are:

- `Teacher_ResNet34_Outputs.zip`
- `Baseline_ResNet18_Outputs.zip`
- `Ternary_KD_QAT_Student_Outputs.zip`
- `Evaluation_Ablation_Outputs.zip`

Because these archives are relatively large, they are provided through GitHub Releases rather than normal repository file uploads.

---

## Requirements

The project is designed to run in a Kaggle notebook environment with GPU acceleration.

The main Python dependencies are:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- Pillow
- tqdm

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Notes

- The CIFAR-10 dataset is loaded from the provided `cifar.zip` archive.
- No additional CIFAR-10 download is required.
- The official CIFAR-10 test set is not used during training.
- Validation accuracy is used for model selection.
- The teacher is frozen during knowledge-distillation training.
- The ternary student uses ternary weights during the forward pass throughout QAT.
- Latent FP32 weights are maintained for optimization.
- The compact student checkpoint stores the ternary representation and required scaling information.
- The final student checkpoint was verified after reloading.
- The reported compression refers to model storage and does not imply an identical inference-speed improvement.

---

## References

1. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778, 2016.
2. G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," *NIPS Deep Learning and Representation Learning Workshop*, 2015.
3. F. Li, B. Zhang, and B. Liu, "Ternary Weight Networks," arXiv:1605.04711, 2016.
4. C. Zhu, S. Han, H. Mao, and W. J. Dally, "Trained Ternary Quantization," *International Conference on Learning Representations (ICLR)*, 2017.
5. Y. Bengio, N. Léonard, and A. Courville, "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation," arXiv:1308.3432, 2013.
6. P. Yin, J. Lyu, S. Zhang, S. Osher, Y. Qi, and J. Xin, "Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets," *International Conference on Learning Representations (ICLR)*, 2019.
