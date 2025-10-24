import cv2
import os
import random
import math
import logging
import albumentations as A
import numpy as np

from albumentations.core.composition import ReplayCompose
from math import ceil, floor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from typing import List, Tuple

from ..utils.data_utils import (
    ratio_gcd,
    ImageInfo,
    BucketManager,
    BucketBatchIndex,
)
from ..utils.utils import largest_size_below_maxarea, resize_large_size_images

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    def __init__(self, batch_size, min_image_size=64, max_image_size=512, min_bucket_size=8):
        """
        Initialize the dataset

        Args:
            batch_size (int): The size of each batch
            min_image_size (int, optional): The minimum size of the images. Defaults to 64.
            max_image_size (int, optional): The maximum size of the images. Defaults to 512.
            min_bucket_size (int, optional): The minimum size of each bucket. Defaults to 8.
        """
        self.imgtotensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.aug_data = ReplayCompose(
            [
                A.HorizontalFlip(p=0.2),
                A.OneOf([
                    A.Rotate(limit=25, interpolation=cv2.INTER_AREA, crop_border=True, p=0.2),
                    A.OneOf([
                        A.RandomCropFromBorders(crop_left=0.05, crop_right=0, crop_top=0, crop_bottom=0, p=1),
                        A.RandomCropFromBorders(crop_left=0, crop_right=0.05, crop_top=0, crop_bottom=0, p=1),
                        A.RandomCropFromBorders(crop_left=0, crop_right=0, crop_top=0.25, crop_bottom=0, p=1),
                        A.RandomCropFromBorders(crop_left=0, crop_right=0, crop_top=0, crop_bottom=0.25, p=1),
                    ], p=0.8)
                ], p=1),
            ],
        )
        self.aug_cond = A.Compose([
            # A.RGBShift(10, 10, 10, p=0.5),
            A.AdvancedBlur(p=0.25),
        ])

        self.data = None
        self._length = 0
        self.batch_size = batch_size
        self.bucket_manager = BucketManager(min_bucket_size)
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.max_area = self.max_image_size * self.max_image_size
        self.downsize_cond = False

    def __len__(self):
        return self._length

    def __getitem__(self, bucket_batch_idx):
        while True:
            try:
                item = self.get_sample(bucket_batch_idx)
            except:
                item = self.get_sample(random.randint(0, self._length - 1))
            return item

    def register_images_and_make_buckets(self, bucket_type='aspect', reso_steps=64, tolerance=0.05):
        assert self.data is not None
        logger.info(f"Registering {len(self.data)} images to bucket manager")
        for info in tqdm(self.data):
            imageinfo = ImageInfo(image_key= info['image_key'], caption=info['caption'], absolute_path=info['absolute_path'])
            if info.get('items') is not None and isinstance(info['items'], str):
                imageinfo.items = info['items']
            if info.get('cond_path') is not None and isinstance(info['cond_path'], str): imageinfo.cond_img_path = info['cond_path']
            imageinfo.ori_size = info['original_size']

            w, h = info['original_size']
            
            if bucket_type == 'aspect':
                max_hw = max(w, h)
                resized_w, resized_h = ceil(w/max_hw*self.max_image_size), ceil(h/max_hw*self.max_image_size)
                resized_w = max(self.min_image_size, resized_w - resized_w % 32)
                resized_h = max(self.min_image_size, resized_h - resized_h % 32)
                
                bucket_guide = ratio_gcd(resized_w, resized_h)
                imageinfo.bucket_reso = (resized_w, resized_h)
                imageinfo.bucket_asp = tuple(bucket_guide)

            elif bucket_type == 'resolution':
                # the logic here is not to upsize small images, but downsize large images
                if w * h > self.max_area:
                    resized_size = largest_size_below_maxarea(w, h, self.max_area, reso_steps)
                else:
                    resized_size = (w, h)
                # resized_size = largest_size_below_maxarea(w, h, self.max_area, reso_steps)

                bucket_width = resized_size[0] - resized_size[0] % reso_steps
                bucket_height = resized_size[1] - resized_size[1] % reso_steps
                if bucket_width == 0 or bucket_height == 0: continue
                    
                bucket_guide = (bucket_width, bucket_height)
                imageinfo.bucket_reso = bucket_guide

            bucket_idx = self.bucket_manager.find_or_create_bucket(bucket_guide, tolerance=tolerance)
            self.bucket_manager.buckets[bucket_idx].append(imageinfo)

        # self.bucket_manager.filter_small_buckets()
        # make sure all bucket images have the same resolution
        for idx in range(len(self.bucket_manager.buckets)):
            bucketimages = self.bucket_manager.buckets[idx]
            bucket_reso = bucketimages[0].bucket_reso
            bucket_asp = bucketimages[0].bucket_asp
            logger.info(f"bucket {idx}: resolution {bucket_reso}, aspect ratio {bucket_asp}, count: {len(bucketimages)}")
            for info in bucketimages[1:]:
                info.bucket_reso = bucket_reso
                info.bucket_asp = bucket_asp


    def shuffle_buckets(self):
        self.bucket_manager.shuffle()

        self.bucket_batch_indices: List[BucketBatchIndex] = []
        for bucket_idx, bucket in self.bucket_manager.buckets.items():
            batch_count = int(ceil(len(bucket) / self.batch_size))
            for bucket_batch_idx in range(batch_count):
                self.bucket_batch_indices.append(BucketBatchIndex(bucket_idx, bucket_batch_idx))

        random.shuffle(self.bucket_batch_indices)
        self._length = len(self.bucket_batch_indices)
        logger.info(f"Total training steps for each epoch is {self._length}")

    def get_sample(self, idx):
        # Implemented for each specific dataset
        raise NotImplementedError