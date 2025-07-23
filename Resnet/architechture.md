
# ResNet-18 Model Architecture (CIFAR-10)

This document describes the ResNet-18 architecture as implemented for CIFAR-10 classification.

## Layer-wise Structure

| Layer              | Output Shape   | Parameters   | Details                        |
|--------------------|---------------|-------------|---------------------------------|
| Input              | 3×32×32       | -           | CIFAR-10 RGB image              |
| Conv2d             | 64×32×32      | 1,792       | k=3×3, s=1, p=1                 |
| BatchNorm2d        | 64×32×32      | 128         |                                 |
| ReLU               | 64×32×32      | -           |                                 |
| Block 1 (2 blocks) | 64×32×32      | 74,112      | 2×[3×3, 64], identity shortcut  |
| Block 2 (2 blocks) | 128×16×16     | 230,528     | 2×[3×3, 128], proj/identity     |
| Block 3 (2 blocks) | 256×8×8       | 919,808     | 2×[3×3, 256], proj/identity     |
| Block 4 (2 blocks) | 512×4×4       | 3,674,624   | 2×[3×3, 512], proj/identity     |
| GlobalAvgPool      | 512           | -           |                                 |
| Linear             | 10            | 5,130       |                                 |
| **Total**          | -             | 11,178,762  |                                 |

## Residual Block (BasicBlock)

Each block contains:
- Conv2d (3×3) → BatchNorm → ReLU → Conv2d (3×3) → BatchNorm
- Shortcut connection (identity or 1×1 projection)
- Output: Add (main + shortcut) → ReLU

**Block diagram:**

```
Input (C×H×W)
   │
 ┌─┴─────────────┐
 │ Conv2d 3×3    │
 │ BatchNorm     │
 │ ReLU          │
 │ Conv2d 3×3    │
 │ BatchNorm     │
 └───────────────┘
   │
 ┌───────────────┐
 │ Shortcut      │ (Identity or 1×1 Conv)
 └───────────────┘
   │
Add (main + shortcut)
   │
ReLU
```

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)


For implementation details, see [`model/resnet.py`](model/resnet.py).