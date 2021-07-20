import imageio


def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        if image_name.endswith('.png'):
            print(image_name)
            frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=1)

    return


def run(gan_type="gan_mnist"):
    base_path = f"./results/{gan_type}/"
    img_path = [f"{base_path}epoch_1.png"]
    img_path += [f"{base_path}epoch_{i}.png" for i in range(20, 310, 40)]
    gif_name = f"./results/{gan_type}.gif"
    create_gif(img_path, gif_name)


def main():
    for t in ["gan_mnist", "gan_anime", "dcgan_mnist", "dcgan_anime", "lsgan_mnist", "lsgan_anime", "cgan_mnist"]:
        run(t)


if __name__ == "__main__":
    main()
