import torch
from model.resnet import resnet18


# Test if ResNet model output shape is correct
def test_resnet_output_shape():
    model = resnet18(3, 10)
    model.eval()
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    assert y.shape == (8, 10), f"Expected output shape (8, 10), got {y.shape}"
