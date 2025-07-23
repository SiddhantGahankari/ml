import torch
from tqdm import tqdm
import torch.nn as nn


# Train the model and return loss/accuracy curves
def train_model(model, train_loader, test_loader, device, epochs=50):
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 45], gamma=0.1
    )

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        # Training loop
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_losses.append(train_loss / total_train)
        train_accs.append(correct_train / total_train * 100)

        # Evaluation loop
        model.eval()
        test_loss, correct_test, total_test = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                correct_test += (outputs.argmax(1) == labels).sum().item()
                total_test += labels.size(0)

        test_losses.append(test_loss / total_test)
        test_accs.append(correct_test / total_test * 100)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%")
        print(f"Test  Loss: {test_losses[-1]:.4f}, Test  Acc: {test_accs[-1]:.2f}%")

        if test_accs[-1] > best_acc:
            best_acc = test_accs[-1]
            torch.save(model.state_dict(), "model_save/best_model.pth")
            print(f"Saved new best model with {test_accs[-1]:.2f}% accuracy.")

    return train_losses, test_losses, train_accs, test_accs
