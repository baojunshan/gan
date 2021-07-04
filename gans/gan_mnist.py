import argparse
import torch
from torch import nn
import numpy as np
import random
import time
import cv2
from torchsummary import summary
from torchvision.utils import save_image

from dataloaders import get_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--lr_g", type=float, default=2e-4)
parser.add_argument("--lr_d", type=float, default=2e-4)
parser.add_argument("--n_step_per_epoch", type=int, default=-1)
parser.add_argument("--n_step_per_d", type=int, default=1)
parser.add_argument("--n_step_per_g", type=int, default=1)
parser.add_argument("--n_epoch_per_generate", type=int, default=20)
parser.add_argument("--image_shape", type=tuple, default=(1, 28, 28))
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--result_path", type=str, default="../results/gan_mnist")
config = parser.parse_args()

print("option".center(60, "-"))
for k, v in vars(config).items():
    print(k.rjust(24), ":", v)
print("-" * 60)

torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)
torch.backends.cudnn.deterministic = True

cuda = config.cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

data_generator = get_dataloader(
    path="../data/mnist",
    shape=config.image_shape,
    name="mnist",
    batch=config.batch,
    n_jobs=1,
    shuffle=True,
)  # 0~255
batch_size = len(data_generator) if config.n_step_per_epoch == -1 else config.n_step_per_epoch


class Generator(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.channel = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]

        self.model = nn.Sequential(
            *self.block(self.latent_dim, 128, normalize=False),
            *self.block(128, 256),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_shape))),
            nn.Tanh(),
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


def generate(generator, latent_dim, n=1):
    width = generator.width
    height = generator.width
    channels = generator.channel

    z = torch.from_numpy(np.random.normal(0, 1, (n**2, latent_dim))).type(torch.FloatTensor)
    z = z.to(device)

    gen_images = generator(z).cpu().detach().numpy().transpose((0, 2, 3, 1))

    concat_images = np.zeros((height * n, width * n, channels))
    for i in range(n):
        for j in range(n):
            concat_images[
                i * height: (i + 1) * height,
                j * width: (j + 1) * width
            ] = gen_images[i * n + j]
    concat_images = (concat_images + 1) / 2 * 255
    concat_images = np.round(concat_images, 0).astype(int)
    return concat_images


generator = Generator(
    latent_dim=config.latent_dim,
    image_shape=config.image_shape,
).to(device)

discriminator = Discriminator(
    image_shape=config.image_shape,
).to(device)

print("Generator structure:")
summary(
    model=generator,
    input_size=(config.latent_dim,),
    device=device
)

print("Discriminator structure:")
summary(
    model=discriminator,
    input_size=config.image_shape,
    device=device
)

loss = nn.BCELoss().to(device)

optimizer_g = torch.optim.Adam(
    params=generator.parameters(),
    lr=config.lr_g,
    betas=(0.5, 0.999)
)
optimizer_d = torch.optim.Adam(
    params=discriminator.parameters(),
    lr=config.lr_d,
    betas=(0.5, 0.999)
)

g_loss_value, d_loss_value = 0, 0

for epoch in range(config.epoch):
    start_time = time.time()
    for i, (image, label) in enumerate(data_generator):
        valid = torch.autograd.Variable(torch.ones(image.shape[0], 1), requires_grad=False).to(device)
        fake = torch.autograd.Variable(torch.zeros(image.shape[0], 1), requires_grad=False).to(device)
        real_images = torch.autograd.Variable(image).to(device)
        z = torch\
            .from_numpy(np.random.normal(0, 1, size=(image.shape[0], config.latent_dim)))\
            .type(torch.FloatTensor)\
            .to(device)

        gen_images = generator(z)

        # --------------------
        # Train discriminator
        # --------------------
        if i % config.n_step_per_d == 0:
            optimizer_d.zero_grad()  # 清零梯度
            real_loss = loss(discriminator(real_images), valid)
            fake_loss = loss(discriminator(gen_images).detach(), fake)  # 不更新生成器
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()  # 计算梯度
            d_loss_value = d_loss.item()
            optimizer_d.step()   # 更新梯度
        # -----------------
        #  Train Generator
        # -----------------
        if i % config.n_step_per_g == 0:
            optimizer_g.zero_grad()
            g_loss = loss(discriminator(gen_images), valid)
            g_loss.backward()
            g_loss_value = g_loss.item()
            optimizer_g.step()

        print("\r", " " * 60, end="")
        print(
            f"\r[Epoch {epoch + 1}/{config.epoch}]",
            f"Batch {i + 1}/{batch_size}",
            f"D loss: {d_loss_value:5f} G loss: {g_loss_value:5f}",
            end=""
        )

        if i >= batch_size - 1:
            break
    print(
        f"\r[Epoch {epoch + 1}/{config.epoch}]",
        f"D loss {d_loss_value:5f} G loss {g_loss_value:5f}",
        f"Time {time.time() - start_time:.2f}"
    )

    if epoch == 0 or (epoch + 1) % config.n_epoch_per_generate == 0:
        eval_image = generate(generator, latent_dim=config.latent_dim, n=10)
        cv2.imwrite(f"{config.result_path}/epoch_{epoch + 1}.png", eval_image)
