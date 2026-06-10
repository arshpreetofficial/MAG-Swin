# MSSA-Swin

Multi-domain Spectral–Spatial–Anatomical Representation Learning for Robust Alzheimer’s Disease Classification.

## Overview

MSSA-Swin is a deep learning framework for Alzheimer’s disease classification using neuroimaging data. It integrates:

1. Spectral encoding using DFT
2. Spatial modeling using Swin Transformer
3. Anatomically guided CNN feature extraction
4. Attention-based feature fusion

## Classification Tasks

- CN vs AD
- MCI vs AD
- CN vs MCI

## Datasets

- ADNI: https://adni.loni.usc.edu
- OASIS: https://www.oasis-brains.org

Datasets must be downloaded by users from official sources.

## Repository Structure


MSSA-Swin/
├── train.py
├── evaluate.py
├── predict.py
├── model.py
├── spectral_encoder.py
├── swin_transformer.py
├── anatomical_branch.py
├── fusion_module.py
├── dataset.py
├── preprocessing.py
├── metrics.py
├── utils.py
├── config.yaml
├── requirements.txt
├── DATASET_README.md

Citation:

If you use MSSA-Swin, the source code, benchmark protocol, or pretrained models in your research, please cite:

Arshpreet Kaur and Jagdeep Kaur.

A Multi-domain Attention-Guided Framework for Alzheimer's Disease Classification.



