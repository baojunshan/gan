import torch
from torch import nn
import numpy as np
import cv2
import os
import time


class Generator(nn.Module):
    def __init__(self, in_features, image_shape, device=None):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.size = self.channels * self.width * self.height
        self.device = device or torch.device("cpu")

        self.linear = nn.Sequential(
            nn.Linear(in_features, self.width * (self.height // 8) * (self.width // 8)),
            nn.BatchNorm1d(self.width * (self.height // 8) * (self.width // 8)),
            nn.ReLU(),
        ).to(self.device)

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.width, self.width // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.width // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.width // 2, self.width // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.width // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.width // 4, self.width // 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.width // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.width // 8, self.channels, 3, stride=1, padding=1),
            nn.Tanh()
        ).to(self.device)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.width, self.height // 8, self.width // 8)
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_shape, device=None):
        super(Discriminator, self).__init__()
        self.channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.size = self.channels * self.width * self.height
        self.device = device or torch.device("cpu")

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ).to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(self.height * self.width, 1),
            # nn.Sigmoid(),
        ).to(self.device)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
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
            k=2,
            p=6,
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
        self.k = k
        self.p = p
        self.epoch = epoch
        self.n_epoch_per_evaluate = n_epoch_per_evaluate
        self.device = device
        self.data = data
        self.gen_input = gen_input
        self.steps_per_epoch = steps_per_epoch or len(self.data)
        self.image_save_path = image_save_path
        if self.image_save_path is not None and not os.path.exists(self.image_save_path):
            os.mkdir(self.image_save_path)

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
                real_images = torch.autograd.Variable(torch.from_numpy(images), requires_grad=True).to(self.device)

                z = torch.from_numpy(
                    np.random.normal(0, 1, (images.shape[0], self.gen_input))
                ).type(torch.FloatTensor).to(self.device)

                self.discriminator.train()
                self.generator.train()
                gen_images = self.generator(z)

                # -----------------
                #  Train discriminator
                # -----------------
                if i % self.n_step_per_discriminator == 0:
                    self.optimizer_d.zero_grad()  # 以前的梯度清空

                    real_validity = self.discriminator(real_images)
                    fake_validity = self.discriminator(gen_images)

                    real_grad_out = torch.autograd.Variable(
                        torch.FloatTensor(real_images.size(0), 1).fill_(1.0),
                        requires_grad=False
                    ).to(self.device)
                    real_grad = torch.autograd.grad(
                        real_validity,
                        real_images,
                        real_grad_out,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (self.p / 2)

                    fake_grad_out = torch.autograd.Variable(
                        torch.FloatTensor(gen_images.size(0), 1).fill_(1.0),
                        requires_grad=False
                    ).to(self.device)
                    fake_grad = torch.autograd.grad(
                        fake_validity,
                        gen_images,
                        fake_grad_out,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (self.p / 2)

                    div_gp = torch.mean(real_grad_norm + fake_grad_norm) * self.k / 2

                    # Adversarial loss
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

                    d_loss.backward()  # 梯度下降
                    self.optimizer_d.step()  # 更新优化器

                # -----------------
                #  Train Generator
                # -----------------
                if i % self.n_step_per_generator == 0:
                    self.optimizer_g.zero_grad()
                    gen_images = self.generator(z)
                    g_loss = -torch.mean(self.discriminator(gen_images))
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
                cv2.imwrite(f"{self.image_save_path}/epoch_{epoch + 1}.png", eval_image)

    def generate(self, n=1):
        width = self.generator.width
        height = self.generator.width
        channels = self.generator.channels

        z = torch.from_numpy(np.random.normal(0, 1, (n ** 2, self.gen_input))).type(torch.FloatTensor).to(self.device)

        self.generator.eval()
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
