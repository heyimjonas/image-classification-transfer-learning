# Image Classification with Transfer Learning

A PyTorch implementation of transfer learning using ResNet18 for image classification on the CIFAR-10 dataset.

## Overview

This project demonstrates how to:
- Use a pre-trained ResNet18 model for transfer learning
- Fine-tune the model on the CIFAR-10 dataset
- Implement training and validation loops
- Save the best performing model

## Requirements

Install the required packages:
```
pip install -r requirements.txt
```

## Usage

```
python transfer_learning.py
```

Optional arguments:

--epochs: Number of training epochs (default: 5)

--batch_size: Batch size for training and validation (default: 64)

--lr: Learning rate (default: 0.001)

--output_model: Path to save the trained model (default: 'transfer_model.pth')

