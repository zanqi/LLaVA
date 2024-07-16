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


def save_dataset(output_dir, train_size, val_size, test_size):
    with h5py.File("armbench512x384_5_all.h5", "r") as f:
        img = f["data"]
        mask = f["mask"]

        # TODO: Select images with at least one object

        train_img = np.array(img[:train_size])
        val_img = np.array(img[train_size : train_size + val_size])
        test_img = np.array(
            img[train_size + val_size : train_size + val_size + test_size]
        )
        train_mask = np.array(mask[:train_size])
        val_mask = np.array(mask[train_size : train_size + val_size])
        test_mask = np.array(
            mask[train_size + val_size : train_size + val_size + test_size]
        )

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

        prep_and_save(train_img, train_mask, train_dir, img_dir)
        prep_and_save(val_img, val_mask, val_dir, img_dir)
        prep_and_save(test_img, test_mask, test_dir, img_dir)


def prep_and_save(img, mask, json_dir, img_dir):
    json_data_list = []
    for i in range(img.shape[0]):
        unique_id = str(uuid.uuid4())
        img_name = unique_id + ".jpg"
        img_path = os.path.join(img_dir, img_name)

        img_i = img[i]
        img_i = img_i.astype(np.uint8)
        img_i = Image.fromarray(img_i)
        img_i.save(img_path)

        answer = []
        for cat in np.unique(mask[i]):
            if cat == 0 or cat == 255:
                continue
            mask_i = mask[i]
            mask_i = mask_i == cat
            bbox = {
                "category_id": int(cat),
                "bbox": [
                    int(np.min(np.where(mask_i)[1])),  # x
                    int(np.min(np.where(mask_i)[0])),  # y
                    int(np.max(np.where(mask_i)[1]))
                    - int(np.min(np.where(mask_i)[1])),  # width
                    int(np.max(np.where(mask_i)[0]))
                    - int(np.min(np.where(mask_i)[0])),  # height
                ],
            }
            answer.append(bbox)

        json_data = {
            "id": unique_id,
            "image": img_name,
            "conversations": [
                {
                    "from": "human",
                    "value": 'What are the objects and their bounding boxes in the image? Output in json format: [{"category_id": 1, "bbox": [x_min, y_min, width, height]}, {"category_id": 2, "bbox": [x_min, y_min, width, height]}, ...]',
                },
                {"from": "gpt", "value": json.dumps(answer)},
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
        "--output_dir",
        type=str,
        help="Directory to save the dataset in",
        default="armbench",
        required=False,
    )
    parser.add_argument(
        "--train_size",
        type=int,
        help="Number of training samples",
        default=10,
        required=False,
    )
    parser.add_argument(
        "--val_size",
        type=int,
        help="Number of validation samples",
        default=5,
        required=False,
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="Number of test samples",
        default=5,
        required=False,
    )
    args = parser.parse_args(arguments)
    output_dir = args.output_dir
    train_size = args.train_size
    val_size = args.val_size
    save_dataset("armbench", 10, 5, 5)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
