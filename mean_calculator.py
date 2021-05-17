import numpy as np
from tqdm import tqdm

from src.datasets import Covid, Mnist, Socofing
from src.utils.configuration import Config


def get_image_sum(image):
    if image.shape[0] == 3:
        return np.sum(image, axis=(1, 2))
    else:
        return np.sum(image)


config = Config(path="configs/custom_trainer/image/cnn_covid/dataset.yaml")

dataset = Covid(config.train)

# config = Config(path='configs/custom_trainer/demo/dataset.yaml')

# dataset = Mnist(config.train)

mean = None
meansq = None
count = 0
for index, sample in tqdm(enumerate(dataset)):
    image = sample["image"].numpy()
    if mean is None:
        mean = get_image_sum(image)
    else:
        mean += get_image_sum(image)

    if meansq is None:
        meansq = get_image_sum(image ** 2)
    else:
        meansq += get_image_sum(image ** 2)
    count += image.shape[1] * image.shape[2]

total_mean = mean / count
total_var = (meansq / count) - (total_mean ** 2)
total_std = np.sqrt(total_var)
print("mean: " + str(total_mean))
print("std: " + str(total_std))
