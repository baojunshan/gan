import numpy as np
import math

from torchsummary import summary
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch

CUDA = True
IMAGE_SHAPE = (1, 28, 28)
LATENT_DIM = 100
LR = 2e-4
EPOCH = 100
BATCH = 1000

CUDA = CUDA and torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        x = self.activation(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_features, image_shape):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.channels = image_shape[0]
        self.width = image_shape[2]
        self.height = image_shape[1]

        self.model = nn.Sequential(
            nn.Linear(in_features, self.channels * self.width * self.height),
            nn.BatchNorm1d(self.channels * self.width * self.height),
            nn.ReLU(inplace=True),
            View(shape=(-1, self.channels, self.width, self.height)),
            ConvBlock(self.channels, self.channels * 6, normalize=False),
            ConvBlock(self.channels * 6, self.channels * 12),
            ConvBlock(self.channels * 12, self.channels * 24),
            ConvBlock(self.channels * 24, self.channels),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.channels = image_shape[0]
        self.width = image_shape[2]
        self.height = image_shape[1]

        self.model = nn.Sequential(
            ConvBlock(self.channels, self.channels * 6, normalize=False),
            ConvBlock(self.channels * 6, self.channels * 12),
            ConvBlock(self.channels * 12, self.channels * 24),
            ConvBlock(self.channels * 24, self.channels),
            View(shape=(-1, self.channels * self.width * self.height)),
            nn.Linear(self.channels * self.width * self.height, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


generator = Generator(in_features=LATENT_DIM, image_shape=IMAGE_SHAPE)
summary(generator, input_size=(LATENT_DIM,))

discriminator = Discriminator(image_shape=IMAGE_SHAPE)
summary(discriminator, input_size=IMAGE_SHAPE)

loss = nn.BCELoss()

if CUDA:
    generator.cuda()
    discriminator.cuda()
    loss.cuda()

optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR)

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=BATCH,
    shuffle=True,
)

for epoch in range(EPOCH):
    for i, (images, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(torch.ones(images.size(0), 1), requires_grad=False)
        fake = Variable(torch.zeros(images.size(0), 1), requires_grad=False)

        # Configure input
        real_images = Variable(images.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------
        z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], LATENT_DIM))))
        gen_images = generator(z)

        optimizer_g.zero_grad()
        g_loss = loss(discriminator(gen_images), valid)
        g_loss.backward()
        optimizer_g.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_d.zero_grad()
        real_loss = loss(discriminator(real_images), valid)
        fake_loss = loss(discriminator(gen_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_d.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, EPOCH, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        if i == len(dataloader) - 1:
            save_image(gen_images.data[:25], f"results/gan_{epoch}.png", nrow=5, normalize=True)
