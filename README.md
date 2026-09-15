# Ternary ResNet on CIFAR-10 with Knowledge Distillation

CIFAR-10 image classification using ResNet knowledge distillation and ternary-weight quantization-aware training (QAT).

This project implements a teacher-student training pipeline in which a full-precision ResNet-34 teacher transfers knowledge to a ResNet-18 student. The student uses ternary internal convolutional and fully connected weights during the forward pass while maintaining latent FP32 weights for optimization.

## Project Overview

The project consists of four stages:

1. **ResNet-34 Teacher**
   - Full-precision ResNet-34 trained from scratch.
   - Used as the teacher network for knowledge distillation.

2. **ResNet-18 FP32 Baseline**
   - Full-precision ResNet-18 trained independently.
   - Provides a baseline for evaluating the effect of quantization and knowledge distillation.

3. **Ternary ResNet-18 Student**
   - ResNet-18 trained using knowledge distillation and ternary-weight QAT.
   - Internal convolutional weights are quantized to:
     \[
     \{-\alpha, 0, +\alpha\}
     \]
   - Latent FP32 weights are maintained during optimization.
   - A straight-through estimator (STE) is used to propagate gradients through the ternary quantization operation.

4. **Evaluation and Ablation**
   - Compares the teacher, FP32 baseline, and ternary student.
   - Reports classification accuracy, parameter count, sparsity, checkpoint size, and compression.
   - Includes weight-distribution visualization and a knowledge-distillation temperature ablation.

## Dataset

The experiments use the **CIFAR-10 dataset**.

The dataset was provided as a `cifar.zip` archive and used directly as the dataset source in the Kaggle environment. The archive contains the CIFAR-10 Python dataset files, including the training batches and official test batch.

**No additional CIFAR-10 download is required.**

The dataset split used in the experiments is:

- Training: 45,000 images
- Validation: 5,000 images
- Official test set: 10,000 images

The official test set is kept separate and is only used for final evaluation.

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
