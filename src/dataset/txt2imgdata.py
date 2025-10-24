import cv2
import numpy as np
import os
import pandas as pd
import math
import torch
import random
import json
import logging

from PIL import Image
from tqdm import tqdm

from .base import BaseDataset
from ..utils.data_utils import crop_or_resize_image
from ..utils.utils import resize_large_size_images

logger = logging.getLogger(__name__)

class Txt2ImgDataset(BaseDataset):
    def __init__(self, data_path, batch_size, min_image_size=64, max_image_size=512, min_bucket_size=8):
        super().__init__(batch_size, min_image_size, max_image_size, min_bucket_size)

        self.custom_instance_prompts = True

        json_data = []
        with open(os.path.join(data_path, 'metadata.jsonl'), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json_data.append(json.loads(line))
        
        self.data = []
        for info in json_data:
            img_key = info['file_name']
            caption = info['prompt']
            img_path = os.path.join(data_path, img_key)
            image = Image.open(img_path).convert('RGB')
            w, h = image.size
            self.data.append(
                {
                    'image_key': img_key,
                    'caption': caption,
                    'absolute_path': img_path, 
                    'original_size': (w, h),
                }
            )


    def get_sample(self, bucket_batch_idx):
        bbinfo = self.bucket_batch_indices[bucket_batch_idx]
        bucket = self.bucket_manager.buckets[bbinfo.bucket_idx]
        bb_idx = bbinfo.batch_idx
        bucket_reso = bucket[0].bucket_reso
        
        output = {}
        images = []
        captions = []
        bsz = min(self.batch_size, len(bucket) - bb_idx)
        for image_info in bucket[bb_idx:bb_idx + bsz]:
            image = Image.open(image_info.absolute_path).convert('RGB')
            image = resize_large_size_images(np.array(image), bucket_reso)
            images.append(self.imgtotensor(image))

            captions.append(image_info.caption)

        output['pixel_values'] = torch.stack(images, dim=0)
        output['caption'] = captions

        return output
