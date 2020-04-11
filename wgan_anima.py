import argparse
import os
import numpy as np
import glob
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch

from DCGAN_generator import Generator
from DCGAN_discriminator import Discriminator

os.makedirs("images_anima", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.001, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
root_dir=r'D:\电子书\TensorFlow2.0深度学习\faces\*.jpg'

class MyDataset():
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_info = glob.glob(self.root_dir)[0:2000]
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info[idx]
        img = Image.open(image_path).convert('RGB')
        sample = {'image': img}
        if self.transform:
            sample['image'] = self.transform(img)
        return sample

origin_dataset= MyDataset(root_dir,transform=transforms.Compose(
            [transforms.Resize(opt.img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]))

dataloader = torch.utils.data.DataLoader(
    dataset=origin_dataset,
    batch_size=opt.batch_size,
    shuffle=True
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader):      # batch id, (image, target)

        # Configure input
        imgs=data['image']
        real_imgs = imgs.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_D.zero_grad()     # 对已有的gradient清零(因为来了新的batch_size的image)

        # Sample noise as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim,2,2)))

        # Generate a batch of images
        fake_imgs = generator(z)     # G(z) ——> D(G(z))

        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs))+torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value,opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            # ------------
            # Train generator
            # ------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs=generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()


    print("[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, opt.n_epochs, loss_D.item(), loss_G.item()))
    if (epoch + 1) % 10 == 0:
        save_image(gen_imgs.data[:25], "images_anima/%d.png" % (epoch+1), nrow=5, normalize=True)

