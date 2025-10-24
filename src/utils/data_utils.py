import cv2
import os
import numpy as np
import random
import torch
import pickle

from collections import defaultdict
from PIL import Image
from math import gcd
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    NamedTuple,
)

import logging
logger = logging.getLogger(__name__)

def ratio_gcd(a: int, b: int) -> int:
    g = gcd(a, b)
    return a//g, b//g

class ImageInfo:
    def __init__(self, image_key: str, caption: str, absolute_path: str) -> None:
        self.image_key: str = image_key
        self.absolute_path: str = absolute_path
        self.caption: str = caption
        self.items: Optional[str] = None
        self.empty_items: Optional[str] = None
        self.image: Optional[Image.Image] = None
        self.ori_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.bucket_asp: Tuple[int, int] = None
        self.bucket_idx: int = None
        self.cond_img_path: str = None


def crop_or_resize_pair_images(image: np.ndarray, cond_image: np.ndarray, bucket_reso: Tuple[int, int], random_crop: bool) -> Tuple[np.ndarray, np.ndarray]:
    image_width, image_height = image.size

    if image_width != bucket_reso[0] or image_height != bucket_reso[1]:
        # image = cv2.resize(image, bucket_reso, interpolation=cv2.INTER_AREA)  # INTER_AREA
        # cond_image = cv2.resize(cond_image, bucket_reso, interpolation=cv2.INTER_AREA)  # INTER_AREA
        image = image.resize(bucket_reso, Image.LANCZOS)
        cond_image = cond_image.resize(bucket_reso, Image.LANCZOS)
    image_width, image_height = image.size

    image = np.array(image)
    cond_image = np.array(cond_image)
    if image_width > bucket_reso[0]:
        trim_size = image_width - bucket_reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        image = image[:, p : p + bucket_reso[0]]
        cond_image = cond_image[:, p : p + bucket_reso[0]]
    elif image_height > bucket_reso[1]:
        trim_size = image_height - bucket_reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        image = image[p : p + bucket_reso[1]]
        cond_image = cond_image[p : p + bucket_reso[1]]
    else:
        pass

    return image, cond_image

def crop_or_resize_image(image: np.ndarray, bucket_reso: Tuple[int, int], random_crop: bool) -> np.ndarray:
    image_height, image_width = image.shape[0:2]

    if image_width != bucket_reso[0] or image_height != bucket_reso[1]:
        image = cv2.resize(image, bucket_reso, interpolation=cv2.INTER_AREA)  # INTER_AREA

    image_height, image_width = image.shape[0:2]

    if image_width > bucket_reso[0]:
        trim_size = image_width - bucket_reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        image = image[:, p : p + bucket_reso[0]]
    elif image_height > bucket_reso[1]:
        trim_size = image_height - bucket_reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        image = image[p : p + bucket_reso[1]]
    else:
        pass

    return image

class BucketBatchIndex(NamedTuple):
    bucket_idx: int
    batch_idx: int

class BucketManager:
    def __init__(self, min_bucket_size: int = 8):
        self.min_bucket_size = min_bucket_size
        self.buckets = defaultdict(list)
        self.bucket_ratios = []

    def shuffle(self):
        for idx, bucket in self.buckets.items():
            random.shuffle(bucket)
        
    def find_or_create_bucket(self, ratio: Tuple[int, int], tolerance=0.1) -> int:
        """Find existing bucket or create new one for given aspect ratio."""
        for idx, bucket_ratio in enumerate(self.bucket_ratios):
            # if abs(bucket_ratio - ratio) <= self.bucket_tolerance:
            if abs(bucket_ratio[0]/bucket_ratio[1] - ratio[0]/ratio[1]) <= tolerance:
                return idx
        
        self.bucket_ratios.append(ratio)
        return len(self.bucket_ratios) - 1
    
    def filter_small_buckets(self):
        """remove buckets smaller than min_bucket_size."""

        new_buckets = defaultdict(list)
        new_ratios = []
        
        dismissed = 0
        for bucket_idx, images in self.buckets.items():
            if len(images) >= self.min_bucket_size:
                new_buckets[len(new_ratios)] = images
                new_ratios.append(self.bucket_ratios[bucket_idx])
            else:
                dismissed += len(images)

        logger.info(f"Dismissed {dismissed} images during buckting, due to minimum bucket size is total batch size {self.min_bucket_size}")

        self.buckets = new_buckets
        self.bucket_ratios = new_ratios


def save_pickle(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
    logger.info(f"Object saved to {filename}")


def load_pickle(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    logger.info(f"Loaded object: {filename}")

    return obj

def debug_dataset(train_dataset):
    logger.info(f"Total dataset length (steps): {len(train_dataset)}")
    output_dir = "./output/test_data"
    os.makedirs(output_dir, exist_ok=True)
    while True:
        logger.info(f"")
        for example in train_dataset:
            batch_idx = random.randint(0, example["pixel_values"].shape[0] - 1)

            im = example["pixel_values"][batch_idx]
            caption = example["caption"][batch_idx]

            if "cond_pixel_values" in example:
                cond_im = example["cond_pixel_values"][batch_idx]
                logger.info(f"current batch size: {example['pixel_values'].shape[0]}, image size: {im.size()}, cond_image size: {cond_im.size()}, caption: {caption}")
            else:
                logger.info(f"current batch size: {example['pixel_values'].shape[0]}, image size: {im.size()}, caption: {caption}")

            im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
            im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)
            cv2.imwrite(f"{output_dir}/tar_img.jpg", im)

            if "cond_pixel_values" in example:
                cond_im = ((cond_im.numpy() + 1.0) * 127.5).astype(np.uint8)
                cond_im = np.transpose(cond_im, (1, 2, 0))  # c,H,W -> H,W,c
                cond_im = cond_im[:, :, ::-1]  # RGB -> BGR (OpenCV)
                cv2.imwrite(f"{output_dir}/cond_img.jpg", cond_im)

            import pdb;pdb.set_trace()