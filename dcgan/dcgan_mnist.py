import numpy as np
import random
import argparse

from torchsummary import summary
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--seed", type=int, default=2021)

config = parser.parse_args()

print("option".center(30, "-"))
for k, v in vars(config).items():
    print(k.rjust(12), ":", v)
print("-" * 30)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(config.seed)

cuda = config.cuda and torch.cuda.is_available()

print("cuda will be used:", cuda)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor




dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=1,
    shuffle=True,
)

for i, _ in dataloader:
    print(i.shape)
    break
exit()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, in_features, image_shape):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.channels = image_shape[0]
        self.init_size = image_shape[1] // 4

        self.l1 = nn.Sequential(nn.Linear(self.in_features, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.channels = image_shape[0]
        self.image_size = image_shape[1]

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),  # 14
            *discriminator_block(16, 32),  # 7
            *discriminator_block(32, 64),  # 4
            *discriminator_block(64, 128),  # 2
        )

        # The height and width of downsampled image
        self.adv_layer = nn.Sequential(nn.Linear(128 * 2 ** 2, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.adv_layer(x)
        return x


generator = Generator(in_features=latent_dim, image_shape=image_shape)
# summary(generator, input_size=(LATENT_DIM,))

print([i for i in generator.named_children()])
exit()

discriminator = Discriminator(image_shape=image_shape)
# summary(discriminator, input_size=IMAGE_SHAPE)

loss = nn.BCELoss()

if cuda:
    generator.cuda()
    discriminator.cuda()
    loss.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(epoch):
    for i, (images, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(torch.ones(images.size(0), 1), requires_grad=False).to("cuda:0" if cuda else "cpu")
        fake = Variable(torch.zeros(images.size(0), 1), requires_grad=False).to("cuda:0" if cuda else "cpu")

        # Configure input
        real_images = Variable(images.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------
        z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], latent_dim))))
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
            % (epoch, epoch, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        if i == len(dataloader) - 1:
            save_image(gen_images.data[:25], f"../results/dcgan/{epoch}.png", nrow=5, normalize=True)
