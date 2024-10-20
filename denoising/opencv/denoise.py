import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os 
from icecream import ic

from PIL import Image
plt.style.use('seaborn')


def denoise_img(input_path, output_path, h=10):
    img = cv.imread(input_path)
    dst = cv.fastNlMeansDenoising(img, None, h, 7, 21)
    im = Image.fromarray(dst)
    im.save(output_path)

split_dirs = ['../../test1_images', '../../train_images', '../../val_images']

for h in [20, 25, 30]:
    ic(h)
    for split in split_dirs:
        ic(split)
        img_dir = os.listdir(split)
        for img in img_dir:
            ic(img)
            img_files = os.listdir(f'{split}/{img}')

            for img_file in img_files:
                img_path = f'{split}/{img}/{img_file}'
                output_file = img_path.split('../')[-1]
                output_path = f'output/h_{h}/{output_file}'
                os.makedirs('/'.join(output_path.split('/')[:-1]), exist_ok=True)
                denoise_img(img_path, output_path, h=h)