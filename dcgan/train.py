import argparse
import torch
from torchsummary import summary

from model import Generator, Discriminator, Trainer
from utils import ImageLoader, show_config, setup_seed

from ast import literal_eval


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--data_path", type=str, default="../data/anime_faces/*.jpg")
parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--lr_g", type=float, default=2e-4)
parser.add_argument("--lr_d", type=float, default=2e-4)
parser.add_argument("--image_shape", type=literal_eval, default='(1, 32, 32)')
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--n_epoch_per_evaluate", type=int, default=20)
parser.add_argument("--data_preload", type=bool, default=True)
parser.add_argument("--n_step_per_d", type=int, default=1)
parser.add_argument("--n_step_per_g", type=int, default=1)
parser.add_argument("--result_path", type=str, default="../results/dcgan_anime")
config = parser.parse_args()
show_config(config)

setup_seed(config.seed)

cuda = config.cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

data_generator = ImageLoader(
    path=config.data_path,
    batch_size=config.batch,
    image_shape=config.image_shape,
    pre_load=config.data_preload,
)

generator = Generator(
    in_features=config.latent_dim,
    image_shape=config.image_shape,
    device=device
)

discriminator = Discriminator(
    image_shape=config.image_shape,
    device=device
)

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


trainer = Trainer(
    generator=generator,
    discriminator=discriminator,
    data=data_generator,
    gen_input=config.latent_dim,
    epoch=config.epoch,
    generator_lr=config.lr_g,
    discriminator_lr=config.lr_d,
    n_step_per_generator=config.n_step_per_g,
    n_step_per_discriminator=config.n_step_per_d,
    n_epoch_per_evaluate=config.n_epoch_per_evaluate,
    image_save_path=config.result_path,
    device=device
)

trainer.train()


