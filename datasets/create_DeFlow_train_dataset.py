import os
import argparse
import sys

from tqdm import tqdm
import numpy as np
import imageio

sys.path.insert(0, '../codes')
from data.util import is_image_file, load_at_multiple_scales


def to_integer(x, eps=1e-8):
    x_int = round(x)
    if abs(x - x_int) < eps:
        return x_int
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-source_dir', required=True, help='path to directory containing HR images')
    parser.add_argument('-target_dir', required=True, help='path to target directory')
    parser.add_argument('-scales', nargs='+',  type=float, default=[1, 4], help='scales to downsample/upsample to')
    args = parser.parse_args()

    # Handle scales
    args.scales = [to_integer(scale) for scale in args.scales]

    source_dir = args.source_dir

    scales = args.scales
    scale_dirs = []
    for scale in scales:
        dir = os.path.join(args.target_dir, f'{scale}x')
        scale_dirs.append(dir)
        os.makedirs(dir, exist_ok=True)

    for fn in tqdm(sorted(os.listdir(source_dir))):
        if not is_image_file(fn):
            continue

        images = load_at_multiple_scales(source_dir+fn, scales=args.scales)

        for img, dir in zip(images, scale_dirs):
            imageio.imwrite(os.path.join(dir,fn), img)
            