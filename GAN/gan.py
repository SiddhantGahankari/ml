# Import necessary libraries for building and training the GAN
import torch 
import torch.nn as nn
import torch.functional as F
import torchvision
from torchvision.datasets import MNIST 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

# Set device to GPU if available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# Define transformations for training and test data (normalize to [-1, 1])
transforms_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) ,(0.5,))
])
transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) ,(0.5,))
])

# Load MNIST dataset with defined transforms
train_dataset = MNIST("datasets/" , train=True , download=True , transform=transforms_train)
test_dataset = MNIST("datasets/" , train=False , transform=transforms_test)

batch_size = 64

# Create DataLoaders for training and testing
train_DataLoader = DataLoader(train_dataset , batch_size=batch_size , shuffle = True , num_workers=4)
test_DataLoader = DataLoader(test_dataset , batch_size=batch_size , shuffle = False , num_workers=4)



# Define the Generator network
class Generator(nn.Module):
    def __init__(self, in_size=100 , out_size = 784):
        super(Generator , self).__init__()
        # First linear layer
        self.l1 = nn.Linear(in_size , 256)
        self.lr = nn.LeakyReLU()
        # Second linear layer and batch normalization
        self.l2 = nn.Linear(256 , 512)
        self.bn = nn.BatchNorm1d(512)
        # Output layer
        self.l3 = nn.Linear(512 , out_size)
        self.out = nn.Tanh()
    def forward(self , x):
        x = self.l1(x)
        x = self.lr(x)
        x = self.l2(x)
        x = self.bn(x)
        x = self.lr(x)
        x = self.l3(x)
        return self.out(x)



# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Flatten input image
            nn.Flatten(), 
            # Fully connected layers with LeakyReLU activations
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


# Initialize Generator and Discriminator models
genModel = Generator().to(device=device)
disModel = Discriminator().to(device=device)

# Set learning rate and beta1 for Adam optimizer
lr = 0.0002
beta1 = 0.5

# Define optimizers for both models
genAdam = optim.Adam(genModel.parameters(), lr=lr, betas=(beta1, 0.999))
disAdam = optim.Adam(disModel.parameters(), lr=lr, betas=(beta1, 0.999))

# Binary Cross Entropy loss for real/fake classification
loss_fn = nn.BCELoss()


# Set latent dimension for generator input
latent_dim = 100

# Number of epochs to train
num_epochs = 22

# Training loop for GAN
for epoch in range(num_epochs):

    running_dis_loss = 0.0
    running_gen_loss = 0.0

    # Iterate over batches of real images
    for i, (real_images, _) in enumerate(train_DataLoader):
        
        batch_size = real_images.shape[0]

        # Zero gradients for discriminator
        disAdam.zero_grad()

        # Prepare real images and labels
        real_images = real_images.to(device)
        real_labels = torch.ones(batch_size, 1, device=device)
        
        # Discriminator output and loss for real images
        dis_real_output = disModel(real_images)
        dis_loss_real = loss_fn(dis_real_output, real_labels)
        
        # Generate fake images and labels
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_images = genModel(noise)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # Discriminator output and loss for fake images
        dis_fake_output = disModel(fake_images.detach())
        dis_loss_fake = loss_fn(dis_fake_output, fake_labels)

        # Total discriminator loss and update
        dis_total_loss = dis_loss_real + dis_loss_fake
        dis_total_loss.backward()
        disAdam.step()
        
        running_dis_loss += dis_total_loss.item()
        
        # Zero gradients for generator
        genAdam.zero_grad()
        
        # Generator loss (tries to fool discriminator)
        dis_output_on_fake = disModel(fake_images)
        gen_loss = loss_fn(dis_output_on_fake, real_labels)
        
        gen_loss.backward()
        genAdam.step()

        running_gen_loss += gen_loss.item()

    # Calculate average losses for the epoch
    avg_dis_loss = running_dis_loss / len(train_DataLoader)
    avg_gen_loss = running_gen_loss / len(train_DataLoader)

    # Print progress
    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Avg Discriminator Loss: {avg_dis_loss:.4f} | "
        f"Avg Generator Loss: {avg_gen_loss:.4f}"
    )



# Set generator to evaluation mode
genModel.eval()

# Generate a sample image from random noise
with torch.no_grad():

    noise = torch.randn(1, latent_dim, device=device)
    
    image_vector = genModel(noise)

# Rescale image vector to [0, 1] for visualization
image_vector = (image_vector + 1) / 2

# Reshape vector to 28x28 image
single_image = image_vector.view(28, 28) 

image_numpy = single_image.cpu().numpy()

# Display generated image
plt.imshow(image_numpy, cmap='gray')
plt.axis("off")
plt.show()


