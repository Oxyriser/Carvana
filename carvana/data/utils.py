import numpy as np
import cv2

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


def grouped(iterable, n):
    return zip(*[iter(iterable)] * n)


def to_coco_rle(rle_mask):
    return {"counts": [int(count) for count in rle_mask.split()], "size": [1280, 1918]}


def get_mask(rle_mask):
    mask = np.zeros(1280 * 1918, dtype=np.uint8)
    rle_mask = rle_mask.split()
    rle_mask_tuples = grouped(rle_mask, 2)
    for offset, count in rle_mask_tuples:
        offset = int(offset) - 1
        count = int(count)
        mask[offset : offset + count] = 1
    return mask.reshape((1280, 1918))


def get_poly(mask):
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(contour, 1, True)
    return pol


def get_carvana_dicts(path, rle_masks):
    dataset_dicts = []
    for idx, (img_name, rle_mask) in rle_masks.iterrows():
        record = {}

        filename = f"{path}/train/{img_name}"

        record["file_name"] = filename
        record["image_id"] = img_name[:-4]
        record["height"] = 1280
        record["width"] = 1918

        mask = get_mask(rle_mask)
        mask_idx = np.argwhere(mask)
        (ymin, xmin), (ymax, xmax) = mask_idx.min(axis=0), mask_idx.max(axis=0)
        record["annotations"] = [
            {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": to_coco_rle(rle_mask),
                "category_id": 0,
            }
        ]
        dataset_dicts.append(record)

    return dataset_dicts


def get_3dcarvana_dicts(path, rle_masks, window=2):
    assert 16 % window == 0, "window should be a divisor of 16"

    dataset_dicts = []
    for idx, imgs in grouped(rle_masks.iterrows(), window):
        record = {}

        filenames = [f"{path}/train/{img.img_name}" for img in imgs]

        record["file_name"] = filenames
        record["image_id"] = filenames[0][:-4]
        record["height"] = 1280
        record["width"] = 1918
        record["depth"] = window

        (ymin, xmin), (ymax, xmax) = (1280, 1918), (0, 0)
        for img in imgs:
            mask = get_mask(img)
            mask_idx = np.argwhere(mask)
            (ymin_, xmin_), (ymax_, xmax_) = mask_idx.min(axis=0), mask_idx.max(axis=0)
            ymin = min(ymin, ymin_)
            xmin = min(xmin, xmin_)
            ymin = max(ymax, ymax_)
            ymin = max(xmax, xmax_)
        record["annotations"] = [
            {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                # As for now the segmentation is not implemented
                # "segmentation": to_coco_rle(rle_mask),
                "segmentation": [],
                "category_id": 0,
            }
        ]
        dataset_dicts.append(record)

    return dataset_dicts
