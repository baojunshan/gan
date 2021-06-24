import argparse

from torchsummary import summary

from model import Generator, Discriminator, Trainer
from utils import ImageLoader, show_config, setup_seed, get_device


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--image_shape", type=tuple, default=(1, 28, 28))
parser.add_argument("--latent_dim", type=int, default=100)
config = parser.parse_args()

show_config(config)

setup_seed(config.seed)

data_generator = ImageLoader(
    path="../data/mnist",
    batch_size=64,
    image_shape=(28, 28),
    gray_scale=True
)

generator = Generator(
    in_features=config.latent_dim,
    image_shape=config.image_shape,
    device=get_device(config.cuda)
)

discriminator = Discriminator(
    image_shape=config.image_shape,
    device=get_device(config.cuda)
)

print("Generator structure:")
summary(
    model=generator,
    input_size=(config.latent_dim,)
)

print("Discriminator structure:")
summary(
    model=discriminator,
    input_size=config.image_shape
)


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


