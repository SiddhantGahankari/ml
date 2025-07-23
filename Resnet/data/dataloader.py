from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


# Returns train and test data loaders for CIFAR-10
def get_loaders(batch_size=256):
    # Data augmentation and normalization for training
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    # Normalization for testing
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    # Load CIFAR-10 datasets
    train_dataset = CIFAR10(
        "datasets/", train=True, download=True, transform=transform_train
    )
    test_dataset = CIFAR10("datasets/", train=False, transform=transform_test)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return train_loader, test_loader
