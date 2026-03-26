# TypiClust & Active Learning Pipeline

This repository contains a complete pipeline for cold-start active learning. It 

---

## Installation & Prerequisites

**Requirement:** Python 3.10.x is highly recommended to avoid dependency conflicts with legacy libraries.

### 1. Install PyTorch (CUDA 12.8)
Install the GPU-enabled version of PyTorch first:
```bash
python -m pip install --upgrade pip setuptools
pip install typing-extensions==4.15.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
### 2. Install Main Dependencies
```bash
pip install -r requirements.txt
```
### File explanations

Embeddings.py: Recomputes embeddings based on the saved SimCLR model and saves them as .npy files.

step2.py: Performs K-Means clustering on the saved embeddings and outputs cluster labels as .npy files.

typiclust.py: The core implementation of the TypiClust algorithm.

typiclust_modification.py: A modified version of the algorithm featuring hybrid sampling and new metric instead of knn mean for typicality.

### Explanations
Core Algorithms
typiclust.py: Standard TypiClust implementation.

typiclust_modification.py: Enhanced/Modified TypiClust implementation.

Evaluation & Comparison
The following scripts provide examples of how to use and evaluate the algorithms, eval files compare original typiclust with random and coreset baselines whie comparison files compare original typiclust with my modification:

Linear Evaluation: linear_eval.py and linear_comparison.py

Fully Supervised Evaluation: fully_supervised_eval.py and FS_comparison.py (includes comparisons for the modified algorithm).

#### For things unrelated to main implentation do
```bash
pip install -r requirements.txt
```
### To recompute embeddings 
```bash
cd .\Unsupervised_Classification\
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
```
### To train flexmatch Manually edit flexmatch.py to set your chosen indices.
```bash
cd .\TorchSSL\
python flexmatch.py --c config/flexmatch/flexmatch_cifar10_10_0.yaml
```
#Troubleshooting & Notes
Paths: Ensure you are in the correct subdirectories (.\Unsupervised_Classification\ or .\TorchSSL\) before running their respective scripts, as they rely on local config paths.

Indices: For FlexMatch, the indices must be updated manually in the script to match the output from the TypiClust selection step.

If you are having issues with torchssl or simclr please go to the official repos and follow the official instructions:
https://github.com/TorchSSL/TorchSSL
https://github.com/wvangansbeke/Unsupervised-Classification

## TypiClust Implementation

### Original Work by the Author
- **`typiclust.py`**  
  - Original implementation of TypiClust based on the paper:  
    **Guy Hacohen, Avihu Dekel, Daphna Weinshall (2022). "Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets."**  
    [arXiv:2202.02794](https://arxiv.org/abs/2202.02794)  
  - The algorithm concept comes from the paper; code is written by the author.

- **`typiclust_modification.py`**  
  - Inspired by ideas from the paper, but the modification is **fully original** and developed by the author.  
  - Both the idea and implementation are the author’s own.

### Third-Party Code Adapted or Used

1. **Unsupervised-Classification**  
   - GitHub: [https://github.com/wvangansbeke/Unsupervised-Classification](https://github.com/wvangansbeke/Unsupervised-Classification)  
   - License: **CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International)**  
   - Used for generating embeddings and clustering workflows. Some files modified for integration with TypiClust.  
   - Non-commercial use only.

2. **TorchSSL**  
   - GitHub: [https://github.com/TorchSSL/TorchSSL](https://github.com/TorchSSL/TorchSSL)  
   - License: **MIT License** (Copyright 2021 TorchSSL)  
   - Most of the code is unchanged from the original repo; minor modifications were made for TypiClust experiments.  
   - MIT license permits redistribution and modification as long as the copyright notice is retained.

---

### Notes
- **TypiClust code** is original, and the modification is fully original though inspired by the paper.  
- Credit the TypiClust paper for the original algorithm concept.  
- Third-party code must retain its original license and copyright.  
- Comply with **CC BY-NC 4.0** for Unsupervised-Classification and **MIT** for TorchSSL.