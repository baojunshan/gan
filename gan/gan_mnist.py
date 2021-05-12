import random
import numpy as np

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
EPOCH = 200
BATCH = 128
SEED = 2020


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(SEED)

CUDA = CUDA and torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

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


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Generator(nn.Module):
    def __init__(self, in_features, image_shape):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.channels = image_shape[0]
        self.width = image_shape[2]
        self.height = image_shape[1]
        self.size = self.channels * self.width * self.height

        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, self.size),
            View(shape=(-1, self.channels, self.width, self.height)),
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
        self.size = self.channels * self.width * self.height

        self.model = nn.Sequential(
            View(shape=(-1, self.size)),
            nn.Linear(self.size, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 1),
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

optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

for epoch in range(EPOCH):
    for i, (images, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(torch.ones(images.size(0), 1), requires_grad=False)
        fake = Variable(torch.zeros(images.size(0), 1), requires_grad=False)

        # Configure input
        real_images = Variable(images.type(Tensor))

        z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], LATENT_DIM))))
        gen_images = generator(z)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        if i % 3 == 0:
            optimizer_d.zero_grad()
            real_loss = loss(discriminator(real_images), valid)
            fake_loss = loss(discriminator(gen_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_g.zero_grad()
        g_loss = loss(discriminator(gen_images), valid)
        g_loss.backward()
        optimizer_g.step()

        print(
            f"[Epoch {epoch}/{EPOCH}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():5f}] [G loss: {g_loss.item():5f}]"
        )

        if i == len(dataloader) - 1:
            save_image(gen_images.data[:25], f"../results/gan/{epoch}.png", nrow=5, normalize=True)
