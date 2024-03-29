python train.py --cuda \
--data_path '../data/mnist_png/training/*/*.png' \
--epoch 300 \
--batch 128 \
--seed 2021 \
--lr_g 0.0001 \
--lr_d 0.0001 \
--image_shape '(1, 32, 32)' \
--latent_dim 100 \
--n_epoch_per_evaluate 20 \
--data_preload true \
--n_step_per_d 1 \
--n_step_per_g 1 \
--result_path '../results/dcgan_mnist'
