#!/usr/bin/env python3

"""Generate the ArmBench dataset 
"""

import argparse
import json
import re
import sys
import uuid
from PIL import Image
import h5py
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from string import ascii_lowercase
import io


def save_dataset(h5_path, output_dir, train_size, val_size, test_size):
    with h5py.File(h5_path) as f:
        images = f["data"]
        masks = f["mask"]

        # get train_size number of images for training with single object
        # i = 0
        # train_img, train_mask, i = get_split(train_size, img, mask, i)
        # val_img, val_mask, i = get_split(val_size, img, mask, i)
        # test_img, test_mask, i = get_split(test_size, img, mask, i)

        img_dir = f"{output_dir}/images"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        train_dir = f"{output_dir}/train"
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        val_dir = f"{output_dir}/validation"
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        test_dir = f"{output_dir}/test"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        prep_and_save(images, masks, train_dir, img_dir, train_size, 0)
        prep_and_save(images, masks, val_dir, img_dir, val_size, train_size)
        prep_and_save(images, masks, test_dir, img_dir, test_size, train_size + val_size)


def annotate_grid(image, grid_size):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')

    image_size = image.size[:2]
    (w, h) = image_size

    for i in range(1, grid_size[0]):
        ax.hlines(h * i / grid_size[0], 0, w,
                  color='black', alpha=0.3, linewidth=1)

    for j in range(1, grid_size[1]):
        ax.vlines(w * j / grid_size[1], 0, h,
                  color='black', alpha=0.3, linewidth=1)

    # for i in range(0, grid_size[0]):
    #     ax.annotate(str(i + 1),
    #                 [w * (i + 0.5) / grid_size[0], 0],
    #                 [w * (i + 0.5) / grid_size[0], -10],
    #                 size=12)

    # for i in range(0, grid_size[0]):
    #     ax.annotate(ascii_lowercase[i],
    #                 [0, h * (i + 0.5) / grid_size[0]],
    #                 [-20, h * (i + 0.5) / grid_size[0]],
    #                 size=12)

    for i in range(0, grid_size[0]):
        for j in range(0, grid_size[1]):
            ax.annotate(str(f"{ascii_lowercase[i]}{grid_size[1] - j}"),
                        [w * (i + 0.5) / grid_size[0], h *
                         (j + 0.5) / grid_size[1]],
                        [w * (i + 0.5) / grid_size[0], h *
                         (j + 0.5) / grid_size[1]],
                        size=10,
                        color='white')

    buf = io.BytesIO()
    fig.savefig(buf, transparent=True, bbox_inches='tight',
                pad_inches=0, format='jpg')
    buf.seek(0)
    # close the figure to prevent it from being displayed
    plt.close(fig)
    return Image.open(buf)

def annotate_visual_prompts(
        obs_image,
        grid_size
):
    """Annotate the visual prompts on the image.
    """
    annotated_image = annotate_grid(
        obs_image,
        grid_size,
    )
    return annotated_image


def annotate_images(img, id, img_dir):
    img = img.resize([512, 512], Image.LANCZOS)
    annotated_img = annotate_visual_prompts(
        img,
        grid_size=[5, 5])
    file_name = f'{id}.jpg'
    annotated_img.save(os.path.join(img_dir, file_name))


def box2grid(bbox, x_grid_size, y_grid_size):
    x_min, y_min, x_max, y_max = bbox
    upper_left_tile_col = chr(x_min // x_grid_size + ord('a'))
    upper_left_tile
    return [x_center, y_center]


def prep_and_save(images, masks, json_dir, img_dir, size, start):
    json_data_list = []

    for i in tqdm(range(start, start + size), desc=json_dir):
        img = Image.fromarray(images[i].astype(np.uint8))
        x_grid_size = img.size[0] // 5
        y_grid_size = img.size[1] // 5
        annotate_images(img, i, img_dir)

        boxes = []
        for cat in np.unique(masks[i]):
            if cat == 0 or cat == 255:
                continue
            mask_i = masks[i]
            mask_i = mask_i == cat
            bbox = [
                int(np.min(np.where(mask_i)[1])),  # x_min
                int(np.min(np.where(mask_i)[0])),  # y_min
                int(np.max(np.where(mask_i)[1])),  # x_max
                int(np.max(np.where(mask_i)[0])),  # y_max
            ]
            grid = box2grid(bbox, x_grid_size, y_grid_size)
            boxes.append(grid)

        if not boxes:
            continue

        json_data = {
            "id": i,
            "image": f"{i}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "Perform object detection on the given image. First, resize the image to 1x1, then answer: what are the bounding boxes of the objects in the image? Output should be in  the format of a list of lists, where each list represents a bounding box: [[x_min, y_min, width, height], [x_min, y_min, width, height], ...] \n\nx_min and y_min are placeholder for the coordinate of the top left corner of an object bounding box, width and height are placeholder for the width and height of the bounding box. \n\nThe x_min, y_min, width and height of the bounding box should be positive decimal numbers between 0 and 1. \n\n \n\nFor example, if there are two objects in the image, the bounding boxes of the objects could be [[0.13, 0.24, 0.37, 0.4], [0.5, 0.55, 0.23, 0.25]].",
                },
                {"from": "gpt", "value": json.dumps(boxes)},
            ],
        }

        json_data_list.append(json_data)

    json_output_path = os.path.join(json_dir, "dataset.json")
    with open(json_output_path, "w") as json_file:
        json.dump(json_data_list, json_file, indent=4)


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--h5",
        type=str,
        help="h5 file containing the dataset",
        default="/gscratch/sciencehub/sebgab/Dev/data/armbench512x384_5_all.h5",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the dataset in",
        default="armbench_grid",
        required=False,
    )
    parser.add_argument(
        "--train_size",
        type=int,
        help="Number of training samples",
        default=600,
        required=False,
    )
    parser.add_argument(
        "--val_size",
        type=int,
        help="Number of validation samples",
        default=100,
        required=False,
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="Number of test samples",
        default=100,
        required=False,
    )
    args = parser.parse_args(arguments)
    h5_path = args.h5
    output_dir = args.output_dir
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    save_dataset(h5_path, output_dir, train_size, val_size, test_size)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
