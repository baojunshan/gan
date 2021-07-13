import numpy as np
import glob
import random
import cv2
import torch


def show_config(config):
    print("option".center(60, "-"))
    for k, v in vars(config).items():
        print(k.rjust(24), ":", v)
    print("-" * 60)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ImageLoader:
    def __init__(self, path, batch_size=64, image_shape=(1, 28, 28), label=False, pre_load=False, random_state=2021):
        self.paths = glob.glob(path)
        self.batch_size = batch_size
        self.label = label
        self.pre_load = pre_load
        self.channel = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.label2idx_ = dict()
        self.idx2label_ = dict()
        random.seed(random_state)

        self.images = None
        if self.pre_load:
            self.images = [cv2.imread(f) for f in self.paths]
        if self.label:
            labels = list(set(p.split("/")[-2] for p in self.paths))
            labels.sort(reverse=False)
            self.idx2label_ = dict(enumerate(labels))
            self.label2idx_ = dict((v, k) for k, v in self.idx2label_.items())

    def preprocess(self, img):
        img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_LINEAR)
        img = img.reshape((self.height, self.width, -1))
        img = img.transpose(2, 0, 1)  # cv2读入的数据为h,w,c，而pytorch需要c,h,w
        img = np.array(img).astype(np.float32) / 255 * 2 - 1
        return img

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __iter__(self):
        data = list()
        while True:
            if self.pre_load:
                random.shuffle(self.images)
            else:
                random.shuffle(self.paths)
            for idx in range(len(self.paths)):
                img = self.images[idx] if self.pre_load else cv2.imread(self.paths[idx])
                img = img if self.channel != 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if self.label:
                    label = self.paths[idx].split()[-2]
                    label_idx = self.label2idx_[label]
                    data.append((self.preprocess(img), label_idx))
                else:
                    data.append(self.preprocess(img))
                if len(data) == self.batch_size:
                    data = np.array(data)
                    yield data
                    data = list()


if __name__ == "__main__":
    loader = ImageLoader(path="../data/mnist_png/training/*/*.png", batch_size=3)
    print(len(loader))
    for i in loader:
        print(i.shape)
        cv2.imwrite("test.png", (i[0].transpose(1, 2, 0) + 1) / 2 * 255)
        cv2.imwrite("test.png", (i[2].transpose(1, 2, 0) + 1) / 2 * 255)
        break
