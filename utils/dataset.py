import math
from os.path import dirname, join

import numpy
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils.util import compute_indices
from utils.util import RandomCutOut, RandomFlip, RandomHSV, RandomRotate
from utils.util import RandomGaussianBlur, RandomRGB2IR, RandomTranslate


class Dataset(data.Dataset):
    def __init__(self,
                 filepath,
                 args, params,
                 augment=True):

        self.augment = augment
        self.stride = params['stride']
        self.num_lms = params['num_lms']
        self.samples = self.load_label(filepath)

        self.flip_index = (numpy.array(params['flip_index']) - 1).tolist()
        self.mean_indices = compute_indices(join(dirname(filepath), 'indices.txt'), params)[0]

        self.num_nb = params['num_nb']
        self.resize = transforms.Resize((args.input_size, args.input_size))
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

        self.input_size = args.input_size
        self.transforms = (RandomHSV(),
                           RandomRotate(),
                           RandomCutOut(),
                           RandomRGB2IR(),
                           RandomTranslate(),
                           RandomFlip(params),
                           RandomGaussianBlur())

    def __getitem__(self, index):
        filename, label = self.samples[index]

        image = self.load_image(filename)
        if not self.augment:
            image = self.resize(image)
            image = self.normalize(image)
            return image, label

        for transform in self.transforms:
            image, label = transform(image, label)

        image = self.normalize(image)
        score = numpy.zeros((self.num_lms,
                             int(self.input_size / self.stride),
                             int(self.input_size / self.stride)))
        offset_x = numpy.zeros((self.num_lms,
                                int(self.input_size / self.stride),
                                int(self.input_size / self.stride)))
        offset_y = numpy.zeros((self.num_lms,
                                int(self.input_size / self.stride),
                                int(self.input_size / self.stride)))
        neighbor_x = numpy.zeros((self.num_nb * self.num_lms,
                                  int(self.input_size / self.stride),
                                  int(self.input_size / self.stride)))
        neighbor_y = numpy.zeros((self.num_nb * self.num_lms,
                                  int(self.input_size / self.stride),
                                  int(self.input_size / self.stride)))

        c, h, w = score.shape
        label = label.reshape(-1, 2)
        assert c == label.shape[0]

        for i in range(c):
            mu_x = int(math.floor(label[i][0] * w))
            mu_y = int(math.floor(label[i][1] * h))
            mu_x = max(0, mu_x)
            mu_y = max(0, mu_y)
            mu_x = min(mu_x, w - 1)
            mu_y = min(mu_y, h - 1)
            score[i, mu_y, mu_x] = 1
            shift_x = label[i][0] * w - mu_x
            shift_y = label[i][1] * h - mu_y
            offset_x[i, mu_y, mu_x] = shift_x
            offset_y[i, mu_y, mu_x] = shift_y

            for j in range(self.num_nb):
                x = label[self.mean_indices[i][j]][0] * w - mu_x
                y = label[self.mean_indices[i][j]][1] * h - mu_y
                neighbor_x[self.num_nb * i + j, mu_y, mu_x] = x
                neighbor_y[self.num_nb * i + j, mu_y, mu_x] = y

        score = torch.from_numpy(score).float()
        offset_x = torch.from_numpy(offset_x).float()
        offset_y = torch.from_numpy(offset_y).float()
        neighbor_x = torch.from_numpy(neighbor_x).float()
        neighbor_y = torch.from_numpy(neighbor_y).float()

        return image, (score, offset_x, offset_y, neighbor_x, neighbor_y)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(filename):
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image

    @staticmethod
    def load_label(filepath):
        with open(filepath, 'r') as f:
            samples = f.readlines()
        samples = [x.strip().split() for x in samples]
        if len(samples[0]) == 1:
            return samples

        new_samples = []
        for sample in samples:
            target = numpy.array([float(x) for x in sample[1:]])
            new_samples.append([sample[0], target])
        return new_samples
