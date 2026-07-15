# CIFAR10 CNN Classifier

A PyTorch CNN that classifies CIFAR10 images into 10 categories, built from raw pixel data through to a properly validated training pipeline.

Built the full pipeline from scratch and then debugged it independently: fixing multiple runtime crashes (typos, undefined variables, an invalid optimizer call), and adding a full train/validation split with best-checkpoint saving instead of blindly training to the last epoch.

## Problem Statement

Given a 32×32 color image from one of 10 object categories (airplane, car, bird, cat, etc.), predict its class — a standard multi-class image classification problem, solved using a custom Convolutional Neural Network.

## Dataset

- CIFAR10 — 60,000 32×32 color images across 10 classes (50,000 train / 10,000 test)
- No missing values or preprocessing needed beyond normalization
- Train set further split 90/10 into train/validation for proper model selection

## Approach

- **Preprocessing** — images converted to tensors and normalized to `[-1, 1]`
- **Data split** — 90% train / 10% validation (from training set), separate held-out test set untouched until final evaluation
- **Modeling** — custom 3-block CNN (`Conv2d → ReLU → MaxPool2d`, channels 3→32→64→128) followed by fully connected layers to 10 output classes
- **Training** — Adam optimizer, `CrossEntropyLoss`, tracked via per-epoch train and validation loss
- **Model selection** — best model checkpoint saved based on lowest validation loss, not just the final epoch
- **Evaluation** — best checkpoint reloaded and evaluated once on the held-out test set

## Key Fixes & Learnings

This project went through a real debugging pass, not just a first-draft run:

- **Multiple crash-causing bugs fixed** — a `kernel_size` typo (written as `kernal_size`) in all three conv layers, a missing `model`/`criterion`/`optimizer` definition, and an invalid `optimizer.stop()` call (should be `optimizer.step()`)
- **`model.forward(x)` replaced with `model(x)`** — calling `.forward()` directly bypasses PyTorch's internal hooks
- **Device handling added** — model and batches now move to GPU when available instead of running CPU-only
- **Validation loop added** — training was originally tracking train loss only, with no way to detect overfitting until final test evaluation. Added a validation split, per-epoch validation loss tracking, and best-model checkpointing
- **Overfitting identified via validation curve** — train loss kept decreasing steadily while validation loss started climbing after a few epochs, confirming the model was memorizing training data. Best checkpoint (lowest validation loss) is now used for final evaluation instead of the last epoch's (overfit) model
- **Comment/print hygiene** — corrected leftover copy-paste comments and print typos ("Evaludate" → "Evaluate", "Total valss" → "Total values")

## Tech Stack

Python, PyTorch, torchvision, pandas, matplotlib

## Project Structure

## How to Run

```bash
git clone https://github.com/Devanshyelne/cifar10-cnn-classifier.git
cd cifar10-cnn-classifier
pip install torch torchvision pandas matplotlib
jupyter notebook CNN_for_CIFAR10.ipynb
```

The CIFAR10 dataset downloads automatically via `torchvision.datasets.CIFAR10` on first run.

## Future Work

- Add BatchNorm and Dropout layers to reduce overfitting further
- Data augmentation (random crop, horizontal flip) to improve generalization
- Deeper architecture / residual connections for higher accuracy
- Learning rate scheduling and early stopping based on validation loss
- Streamlit demo for live image classification

Built by Devansh Yelne — 2nd year AI & ML student, learning ML by building and breaking things properly.
