import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

# Plot training and testing loss/accuracy curves and save them
def plot_curves(train_losses, test_losses, train_accs, test_accs, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Plot and save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, test_losses, label="Test Loss", color="red")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    loss_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()  # avoid displaying in some environments

    # Plot and save accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy", color="green")
    plt.plot(epochs, test_accs, label="Test Accuracy", color="orange")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(save_dir, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()

    print(f"Plots saved to '{save_dir}/'")



# Plot and save the confusion matrix
def plot_confusion_matrix(model, dataloader, class_names, save_path="plots/confusion_matrix.png", device=None):
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved to: {save_path}")
