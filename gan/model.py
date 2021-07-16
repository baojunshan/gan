import os
import torch
from torch import nn
import numpy as np
import cv2
import time


class Generator(nn.Module):
    def __init__(self, in_features, image_shape, device=None):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.channels = int(image_shape[0])
        self.height = int(image_shape[1])
        self.width = int(image_shape[2])
        self.size = self.channels * self.width * self.height
        self.device = device or torch.device("cpu")

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
        ).to(self.device)
        self.activate = nn.Tanh()

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.channels, self.width, self.height)
        x = self.activate(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_shape, device=None):
        super(Discriminator, self).__init__()
        self.channels = int(image_shape[0])
        self.height = int(image_shape[1])
        self.width = int(image_shape[2])
        self.size = self.channels * self.width * self.height
        self.device = device or torch.device("cpu")

        self.model = nn.Sequential(
            nn.Linear(self.size, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(self.device)

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
        steps_per_epoch=None,
        epoch=30,
        n_step_per_generator=1,
        n_step_per_discriminator=1,
        generator_lr=1e-4,
        discriminator_lr=1e-4,
        device=torch.device("cpu"),
        n_epoch_per_evaluate=10,
        image_save_path=None
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.n_step_per_generator = n_step_per_generator
        self.n_step_per_discriminator = n_step_per_discriminator
        self.epoch = epoch
        self.n_epoch_per_evaluate = n_epoch_per_evaluate
        self.device = device
        self.data = data
        self.gen_input = gen_input
        self.steps_per_epoch = steps_per_epoch or len(self.data)
        self.image_save_path = image_save_path
        if self.image_save_path is not None and not os.path.exists(self.image_save_path):
            os.mkdir(self.image_save_path)

        self.loss = nn.BCELoss().to(self.device)

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

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
            start_time = time.time()
            for i, images in enumerate(self.data):
                valid = torch.autograd.Variable(torch.ones(images.shape[0], 1), requires_grad=False).to(self.device)
                fake = torch.autograd.Variable(torch.zeros(images.shape[0], 1), requires_grad=False).to(self.device)

                real_images = torch.autograd.Variable(torch.from_numpy(images)).to(self.device)
                z = torch.from_numpy(np.random.normal(0, 1, (images.shape[0], self.gen_input))).type(torch.FloatTensor).to(self.device)

                self.discriminator.train()
                self.generator.train()
                gen_images = self.generator(z)

                # -----------------
                #  Train discriminator
                # -----------------
                if i % self.n_step_per_discriminator == 0:
                    self.optimizer_d.zero_grad()  # 以前的梯度清空

                    real_loss = self.loss(self.discriminator(real_images), valid)
                    fake_loss = self.loss(self.discriminator(gen_images.detach()), fake)  # 不更新生成器
                    d_loss = (real_loss + fake_loss) / 2

                    d_loss.backward()  # 梯度下降
                    self.optimizer_d.step()   # 更新优化器

                # -----------------
                #  Train Generator
                # -----------------
                if i % self.n_step_per_generator == 0:
                    self.optimizer_g.zero_grad()
                    g_loss = self.loss(self.discriminator(gen_images), valid)
                    g_loss.backward()
                    self.optimizer_g.step()

                # print("\r", " " * 60, end="")
                print(
                    f"\r[Epoch {epoch + 1:03}/{self.epoch:03}]",
                    f"Batch {i + 1:05}/{self.steps_per_epoch:05}",
                    f"D loss: {d_loss.item():.5f} G loss: {g_loss.item():.5f}",
                    end=""
                )

                if i >= self.steps_per_epoch - 1:
                    break
            print(f"\r" + " " * 70, end="")
            print(
                f"\r[Epoch {epoch + 1}/{self.epoch}]",
                f"D loss {d_loss.item():5f} G loss {g_loss.item():5f}",
                f"Time {time.time() - start_time:.2f}"
            )

            if (epoch == 0 or (epoch + 1) % self.n_epoch_per_evaluate == 0) and self.image_save_path:
                eval_image = self.generate(n=10)
                cv2.imwrite(f"{self.image_save_path}/epoch_{epoch+1}.png", eval_image)

    def generate(self, n=1):
        width = self.generator.width
        height = self.generator.width
        channels = self.generator.channels

        z = torch.from_numpy(np.random.normal(0, 1, (n**2, self.gen_input))).type(torch.FloatTensor)
        z = z.to(self.device)

        self.generator.train()
        gen_images = self.generator(z).cpu().detach().numpy().transpose((0, 2, 3, 1))

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


if __name__ == "__main__":
    data = np.random.normal(0, 1, (2**2, 10))
    print(data)
