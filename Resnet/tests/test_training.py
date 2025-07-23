import torch
import torch.multiprocessing as mp
from model.resnet import resnet18
from data.dataloader import get_loaders
from train.trainer import train_model


# Test if training runs without errors
def test_training_runs():
    mp.set_start_method("spawn", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(3, 10).to(device)
    train_loader, test_loader = get_loaders(batch_size=32)

    train_model(model, train_loader, test_loader, device, epochs=1)
    # Just checking it runs without crashing
