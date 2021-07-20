python train.py --cuda \
--data_path '../data/anime_faces/*.jpg' \
--epoch 300 \
--batch 128 \
--seed 2021 \
--lr_g 0.0002 \
--lr_d 0.0002 \
--image_shape '(3, 64, 64)' \
--latent_dim 100 \
--n_epoch_per_evaluate 20 \
--data_preload true \
--n_step_per_d 1 \
--n_step_per_g 1 \
--result_path '../results/lsgan_anime'
