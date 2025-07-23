import pytest
from data.dataloader import get_loaders


# Test if dataloader returns correct shapes
def test_dataloader_shapes():
    train_loader, test_loader = get_loaders(batch_size=16)
    images, labels = next(iter(train_loader))

    assert images.shape == (16, 3, 32, 32), "Train images shape incorrect"
    assert labels.shape == (16,), "Train labels shape incorrect"
