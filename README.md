# Adaptive-SpikeRain

Adaptive Spike-Native Spiking Neural Network for Energy-Efficient Image Deraining.

## Research Objective

This project aims to develop an adaptive Spiking Neural Network (SNN) framework for image deraining.

The proposed framework focuses on:

- Rain complexity estimation
- Adaptive temporal processing
- Spike-native feature extraction
- Spike-based attention
- Energy-aware optimization


## Proposed Architecture

Rain Image

↓

Rain Complexity Estimator

↓

Adaptive Temporal Controller

↓

Spike-Native Encoder

↓

Spike Feature Extraction Backbone

↓

Spike Attention Module

↓

Spike Decoder

↓

Clean Image


## Research Gaps

1. Fixed time steps in SNN-based deraining
2. Traditional operations inside SNN
3. Lack of spike-based attention
4. Lack of energy-aware optimization


## Dataset

The model will be evaluated on:

- Rain100L
- Rain100H
- Rain1200
- Rain12


## Framework

- PyTorch
- SpikingJelly
- Kaggle GPU