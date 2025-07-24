import torch 
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
print(device)

transform = transforms.Compose(
    [ transforms.ToTensor() , transforms.Normalize(0.5,0.5)]
)
batch_size = 64

train_dataset = MNIST("dataset/" ,train=True , download=True , transform=transform)
test_dataset = MNIST("dataset/" ,train=False , transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size ,num_workers=4 , shuffle=True)
test_loader = DataLoader(test_dataset , batch_size=batch_size , num_workers=4)

class Generator(nn.Module):
    def __init__(self, in_channels = 100 , out_channels = 28*28):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels , out_channels=128 , kernel_size=7 , stride=1 , padding=0), #128 x 7 x 7,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128 , out_channels=64 , stride=2 ,kernel_size=4 , padding=1), #64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64 , out_channels=1 ,kernel_size=4, stride= 2 , padding=1),     # 1x 28 x 28
            nn.Tanh()
        )
    def forward(self ,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x;


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            # Layer 1: 1 × 28 × 28 → 64 × 14 × 14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.l2 = nn.Sequential(
            # Layer 2: 64 × 14 × 14 → 128 × 7 × 7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.l3 = nn.Sequential(
            # Layer 3: 128 × 7 × 7 → 256 × 3 × 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.l4 = nn.Sequential(
            # Final classification: 256 × 3 × 3 → 1
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x


loss_fn = nn.BCELoss()
latent_dim = 100
genModel = Generator(latent_dim).to(device)
disModel = Discriminator().to(device)

optimizer_gen = torch.optim.Adam(genModel.parameters() , lr = 0.003 , betas= ((0.5 ,0.999)))
optimizer_dis = torch.optim.Adam(disModel.parameters() , lr = 0.001 , betas= ((0.5 ,0.999)))
num_epochs = 50


step = 0
fixed_noise = torch.randn((64, latent_dim, 1, 1)).to(device)

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.to(device)
        batch_size = real.size(0)

        # === Train Discriminator ===
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake = genModel(noise)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        dis_real_output = disModel(real).view(-1, 1)
        loss_real = loss_fn(dis_real_output, real_labels)

        dis_fake_output = disModel(fake.detach()).view(-1, 1)
        loss_fake = loss_fn(dis_fake_output, fake_labels)

        loss_dis = (loss_real + loss_fake) / 2

        disModel.zero_grad()
        loss_dis.backward()
        optimizer_dis.step()

        # === Train Generator ===
        gen_output = disModel(fake).view(-1, 1)
        loss_gen = loss_fn(gen_output, real_labels)

        genModel.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        # === Logging ===
        if batch_idx == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} "
                  f"Loss D: {loss_dis:.4f}, Loss G: {loss_gen:.4f}")

            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    f_img = genModel(fixed_noise)
                    grid = vutils.make_grid(f_img, nrow=8, normalize=True)
                    plt.figure(figsize=(8, 8))
                    plt.axis("off")
                    plt.title(f"Generated Images - Epoch {epoch+1}")
                    plt.imshow(grid.permute(1, 2, 0).cpu())
                    plt.show()





