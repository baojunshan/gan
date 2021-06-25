import numpy as np
import glob
import random
import cv2
import torch


def get_device(cuda=True):
    flag = cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if flag else "cpu")
    return device


def show_config(config):
    print("option".center(30, "-"))
    for k, v in vars(config).items():
        print(k.rjust(12), ":", v)
    print("-" * 30)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ImageLoader:
    def __init__(self, path, batch_size=64, image_shape=(28, 28), gray_scale=True, random_state=2021, load_all=False):
        self.paths = glob.glob(f"{path}/*")
        self.batch_size = batch_size
        self.height = image_shape[0]
        self.width = image_shape[1]
        self.gray_scale = gray_scale
        self.load_all = load_all
        random.seed(random_state)

        self.data = list() if not self.load_all else [self._load(p) for p in self.paths]

    def _load(self, path):
        if self.gray_scale:
            img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.imread(path)

        img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_LINEAR)
        img = img.reshape((self.height, self.width, -1))
        img = img.transpose(2, 0, 1)  # cv2读入的数据为h,w,c，而pytorch需要c,h,w
        img = img.astype(np.float32) / 255 * 2 - 1
        return img

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __iter__(self):
        data = list()
        while True:
            if self.load_all:
                random.shuffle(self.data)
                for d in self.data:
                    data.append(d)
                    if len(data) == self.batch_size:
                        data = np.array(data)
                        yield data
                        data = list()
            else:
                random.shuffle(self.paths)
                for f in self.paths:
                    img = self._load(f)
                    data.append(img)
                    if len(data) == self.batch_size:
                        data = np.array(data)
                        yield data
                        data = list()


if __name__ == "__main__":
    loader = ImageLoader(path="../data/img", batch_size=1)
    for i in loader:
        print(i.shape)
        cv2.imwrite("test.png", (i[0].transpose(1, 2, 0) + 1) / 2 * 255)
        break
