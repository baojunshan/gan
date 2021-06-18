import torch
from torch import nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, in_features, image_shape):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
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
        )
        self.activate = nn.Tanh()

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.channels, self.width, self.height)
        x = self.activate(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.size = self.channels * self.width * self.height

        self.model = nn.Sequential(
            nn.Linear(self.size, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.size)
        x = self.model(x)
        return x


class Trainer:
    def __init__(
        self,
        generator,
        discriminator,
        data,
        gen_input,
        steps_per_epoch=100,
        epoch=30,
        generator_n_per_step=1,
        discriminator_n_per_step=1,
        generator_lr=1e-4,
        discriminator_lr=1e-4,
        device=torch.device("cpu"),
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.generator_n_per_step = generator_n_per_step
        self.discriminator_n_per_step = discriminator_n_per_step
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.device = device
        self.data = data
        self.gen_input = gen_input

        self.loss = nn.BCELoss()

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.loss.to(self.device)

        self.optimizer_g = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=self.generator_lr,
            betas=(0.5, 0.999)
        )
        self.optimizer_d = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=self.discriminator_lr,
            betas=(0.5, 0.999)
        )

    def train(self):
        for epoch in range(self.epoch):
            for i, images in enumerate(self.data):
                valid = torch.autograd.Variable(torch.ones(images.shape[0], 1), requires_grad=False)
                fake = torch.autograd.Variable(torch.zeros(images.shape[0], 1), requires_grad=False)

                real_images = torch.autograd.Variable(images.type(torch.FloatTensor))
                real_images.to(self.device)

                z = torch.autograd.Variable(np.random.normal(0, 1, (images.shape[0], self.gen_input)))
                z.to(self.device)

                gen_images = self.generator(z)

                # -----------------
                #  Train discriminator
                # -----------------
                if i % self.discriminator_n_per_step == 0:
                    self.optimizer_d.zero_grad()  # 以前的梯度清空

                    real_loss = self.loss(self.discriminator(real_images), valid)
                    fake_loss = self.loss(self.discriminator(gen_images.detach()), fake)  # 不更新生成器
                    d_loss = (real_loss + fake_loss) / 2

                    d_loss.backward()  # 梯度下降
                    self.optimizer_d.step()   # 更新优化器

                # -----------------
                #  Train Generator
                # -----------------
                if i % self.generator_n_per_step == 0:
                    self.optimizer_g.zero_grad()
                    g_loss = self.loss(self.discriminator(gen_images), valid)
                    g_loss.backward()
                    self.optimizer_g.step()

                print(
                    f"[Epoch {epoch}/{self.epoch}] [Batch {i}/{self.steps_per_epoch}]" +
                    f"[D loss: {d_loss.item():5f}] [G loss: {g_loss.item():5f}]"
                )

            if i == len(dataloader) - 1:
                    save_image(gen_images.data[:25], f"../results/gan/{epoch}.png", nrow=5, normalize=True)

    def predict(self):
        pass
