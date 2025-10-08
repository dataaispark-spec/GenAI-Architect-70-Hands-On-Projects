# Course 15: Capstone: Basic GAN - Hands-on Labs 🎯🖌️

## Lab Overview: Building Your First GAN Image Generator

This capstone lab crowns your AI journey by constructing a complete Generative Adversarial Network that creates custom images from scratch. You'll experience the thrill of GAN training, debug common failures, and emerge ready to architect generative systems in the real world.

---

## Prerequisites
- Google Colab Pro (for GPU access)
- PyTorch installation
- Custom dataset (images) uploaded
- Patience for iterative training!

## 🎨 **Lab Objectives**
- Implement DCGAN architecture in PyTorch
- Train on custom image datasets
- Debug and iterate through GAN training issues
- Deploy trained model for inference
- Generate high-quality synthetic images

## 📋 **Step-by-Step Implementation**

### Step 1: Environment Setup 🌐
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

### Step 2: Custom Dataset Preparation 📁
```python
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image

# Prepare transforms for 64x64 images (DCGAN standard)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load your custom dataset
dataset = CustomImageDataset('/path/to/your/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

# Visualize sample images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Real Images from Custom Dataset")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()
```

### Step 3: DCGAN Architecture Implementation 🏗️
```python
# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # Final layer - output is 3x64x64 (RGB image)
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is 3x64x64 (RGB image)
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer - output probability
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Initialize networks
netG = Generator().to(device)
netD = Discriminator().to(device)

print(netG)
print(netD)
```

### Step 4: Loss Functions and Optimizers ⚙️
```python
# Loss functions
criterion = nn.BCELoss()

# Create batch of latent vectors for visualization
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

# Establish convention for real and fake labels
real_label = 1.0
fake_label = 0.0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training tracking
img_list = []
G_losses = []
D_losses = []
iters = 0
```

### Step 5: Training Loop 🎯
```python
num_epochs = 25

print("Starting Training Loop...")
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

        iters += 1

print('Training complete!')
```

### Step 6: Visualization and Evaluation 🎨
```python
# Plot training losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Show generated images progression
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
plt.show()

# Final generated images
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[:64].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
```

### Step 7: Iteration on Failures 🔄
**Common Issues & Fixes:**

```python
# Issue 1: Mode Collapse - Generator produces same image
# Fix: Use mini-batch discrimination or unrolled GAN
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims[0], kernel_dims[1]))

    def forward(self, x):
        # Implementation for batch discrimination
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims[0], self.kernel_dims[1])

        M = matrices.unsqueeze(0)  # 1 x b x out x kernel
        M_T = M.permute(1, 0, 2, 3)  # b x 1 x out x kernel
        norm = torch.abs(M - M_T).sum(3)  # b x b x out
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # b x out, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x

# Issue 2: Training Instability - Oscillating losses
# Fix: Implement gradient penalty (WGAN-GP)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.randn((real_samples.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones((real_samples.shape[0], 1), device=device, requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Modify training loop for better stability
lambda_gp = 10  # Gradient penalty coefficient

# In discriminator update:
gradient_penalty = compute_gradient_penalty(netD, real_cpu, fake.detach())
errD = errD_real + errD_fake + lambda_gp * gradient_penalty

# Issue 3: Poor Generated Quality
# Fix: Use Wasserstein distance instead of BCE

# Replace BCELoss with:
def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)

# Replace optimizer with RMSprop for WGAN
optimizerD = optim.RMSprop(netD.parameters(), lr=0.00005)
optimizerG = optim.RMSprop(netG.parameters(), lr=0.00005)

# Remove sigmoid from discriminator output
# Training becomes: maximize D(real) - D(fake)
```

### Step 8: Model Saving and Inference 🚀
```python
# Save the trained model
torch.save({
    'generator_state_dict': netG.state_dict(),
    'discriminator_state_dict': netD.state_dict(),
    'optimizerG_state_dict': optimizerG.state_dict(),
    'optimizerD_state_dict': optimizerD.state_dict(),
}, 'dcgan_model.pth')

# Load and generate new images
checkpoint = torch.load('dcgan_model.pth')
netG.load_state_dict(checkpoint['generator_state_dict'])

# Generate new images
with torch.no_grad():
    fake_images = netG(torch.randn(10, 100, 1, 1, device=device)).detach().cpu()
    
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(np.transpose(fake_images[i], (1, 2, 0)) * 0.5 + 0.5)  # Denormalize
    plt.axis('off')
plt.show()
```

---

## 🎯 **Questions & Iterations**

### Checkpoint Assessment
1. **Does discriminator dominate too early?** Increase learning rate for G
2. **Mode collapse detected?** Add mini-batch discrimination
3. **Generator not converging?** Check if discriminator is too strong

### Advanced Iterations
1. **Implement WGAN-GP** for stable training
2. **Add conditioning** for controlled generation
3. **Scale to higher resolution** with progressive growing
4. **Apply to different domains** (faces, art, medical)

### Success Metrics
- **FID Score < 50** indicates reasonable image quality
- ** discriminator loss around 0.5** at convergence
- **Visual coherence** in generated samples

---

## 📚 **Extended Resources**
- **DCGAN Paper**: Radford et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
- **WGAN Paper**: Arjovsky et al. "Wasserstein GAN"
- **PyTorch GAN Implementations**: pytorch.org/tutorials intermediates/dcgan_faces_tutorial.html

---

## 🏆 **Capstone Achievement**
Building this GAN demonstrates mastery of:
- PyTorch custom model development
- Adversarial training dynamics
- Debugging complex ML systems
- Iterative problem-solving in AI

**You're now equipped to architect generative systems that create new realities! 🎨✨**

*Congratulations on completing your GenAI Architect journey.*
