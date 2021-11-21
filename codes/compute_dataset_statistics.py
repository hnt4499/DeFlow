import os
import argparse

from tqdm import tqdm
import numpy as np

from data import imread


accepted_ext = [".png", ".jpg"]


def main(args):
    data_dir = args.data_dir
    images = []

    for filename in tqdm(os.listdir(data_dir)):
        _, ext = os.path.splitext(filename)
        if ext in accepted_ext:
            filepath = os.path.join(data_dir, filename)
            image = imread(filepath)
            images.append(image.reshape(image.shape[0], -1))

    if len(images) == 0:
        raise RuntimeError("No images found.")

    images = np.concatenate(images, axis=1)
    mean = images.mean(axis=1)
    std = images.std(axis=1)
    print(f"Mean: {mean}, std: {std}")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type=str, help="Path to the directory containing all images")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)