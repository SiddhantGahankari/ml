import torch
from model.resnet import resnet18
from data.dataloader import get_loaders
from train.trainer import train_model
from visualize.plots import plot_curves
from torchinfo import summary

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CIFAR-10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Initialize ResNet-18 model
model = resnet18(3, 10).to(device)
summary(model, input_size=(1, 3, 32, 32))

# Get data loaders
train_loader, test_loader = get_loaders()
# Train the model
train_losses, test_losses, train_accs, test_accs = train_model(
    model, train_loader, test_loader, device, epochs=100
)
# Save the trained model
torch.save(model.state_dict(), "model_save/final_model.pth")

# Plot training curves
plot_curves(train_losses, test_losses, train_accs, test_accs)
