from PIL import Image
import random
import os
from config import *



def image_generator(a_path, b_path, batch_size, shuffle=True):
    image_filenames = os.listdir(a_path)
    n_batch = len(image_filenames) / batch_size if len(image_filenames) % batch_size == 0 else len(
        image_filenames) / batch_size + 1
    while True:
        if shuffle: random.shuffle(image_filenames)
        for i in range(n_batch):
            a_batch = []
            b_batch = []
            for j in range(batch_size):
                index = i * batch_size + j
                if index >= len(image_filenames): continue
                a = Image.open(os.path.join(a_path, image_filenames[index])).convert('RGB')
                b = Image.open(os.path.join(b_path, image_filenames[index])).convert('RGB')
                a = a.resize((crop_from, crop_from), Image.BICUBIC)
                b = b.resize((crop_from, crop_from), Image.BICUBIC)
                a = np.asarray(a) / 127.5-1
                b = np.asarray(b) / 127.5-1
                w_offset = np.random.randint(0, max(0, crop_from - image_size - 1)) if shuffle else (crop_from - image_size)/2
                h_offset = np.random.randint(0, max(0, crop_from - image_size - 1)) if shuffle else (crop_from - image_size)/2
                a = a[h_offset:h_offset + image_size, w_offset:w_offset + image_size, :]
                b = b[h_offset:h_offset + image_size, w_offset:w_offset + image_size, :]
                a_batch.append(a)
                b_batch.append(b)
            if direction == 'a2b':
                yield np.array(a_batch), np.array(b_batch)
            else:
                yield np.array(b_batch), np.array(a_batch)
            #yield (np.array(a_batch) - imagenet_mean) / imagenet_std, (np.array(b_batch) - imagenet_mean) / imagenet_std
