from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(path, shape, name=None, batch=64, n_jobs=1, shuffle=True):
    if isinstance(shape, int):
        shape = (shape, shape)
    elif len(shape) == 3:
        shape = (shape[1], shape[2])
    elif len(shape) == 1:
        shape = (shape[0], shape[0])

    transform_compose = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor()
    ])

    if name == "mnist":
        dataset = datasets.MNIST(root=path, train=True, download=True, transform=transform_compose)
    elif name == "fashion":
        dataset = datasets.FashionMNIST(root=path, train=True, download=True, transform=transform_compose)
    elif name == "lsun":
        dataset = datasets.LSUN(root=path, transform=transform_compose)
    else:
        dataset = datasets.ImageFolder(root=path, transform=transform_compose)

    return DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle,
        # num_workers=n_jobs,
    )


if __name__ == "__main__":
    dataloader = get_dataloader(path="../data", shape=(28, 28), name="mnist")
    for i in dataloader:
        print(i.shape)
        break
