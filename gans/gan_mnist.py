import argparse
import torch
from torch import nn
import numpy as np
import random
import cv2
from torchsummary import summary

from dataloaders import get_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--image_shape", type=tuple, default=(1, 28, 28))
parser.add_argument("--latent_dim", type=int, default=100)
config = parser.parse_args()

print("option".center(30, "-"))
for k, v in vars(config).items():
    print(k.rjust(12), ":", v)
print("-" * 30)

torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)
torch.backends.cudnn.deterministic = True

cuda = config.cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

print(device, torch.cuda.is_available())

data_generator = get_dataloader(
    path="../data/mnist",
    shape=config.image_shape,
    name="mnist",
    batch=config.batch,
    n_jobs=1,
    shuffle=True,
)

class Generator(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape

        self.model = nn.Sequential(
            *self.block(self.latent_dim, 128, normalize=False),
            *self.block(128, 256),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_shape))),
            nn.Tanh()
        )

    @staticmethod
    def block(in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, inputs):
        img = self.model(inputs)
        img = img.view(img.size(0), *self.image_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.image_shape = image_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        img_flat = inputs.view(inputs.size(0), -1)
        validity = self.model(img_flat)
        return validity


generator = Generator(
    latent_dim=config.latent_dim,
    image_shape=config.image_shape,
)

discriminator = Discriminator(
    image_shape=config.image_shape,
)

# print("Generator structure:")
# summary(
#     model=generator,
#     input_size=(config.latent_dim,)
# )
#
# print("Discriminator structure:")
# summary(
#     model=discriminator,
#     input_size=config.image_shape
# )
exit()
batch_size = min(config.batch, len(data_generator))

for epoch in config.epoch:
    for batch, (data, label) in enumerate(data_generator):
        if batch >= batch_size:
            break



trainer = Trainer(
    generator=generator,
    discriminator=discriminator,
    data=data_generator,
    gen_input=config.latent_dim,
    epoch=2000,
    n_epoch_per_evaluate=200,
    # log_path="logs/gan_mnist",
    image_save_path="../results/gan/mnist",
    device=get_device(config.cuda)
)

trainer.train()


