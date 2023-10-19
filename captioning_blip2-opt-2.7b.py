# This is a sample Python script.

import os
import argparse
import random
import json
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument(
        '--save_path',
        default='../datasets/captions/',
        type=str,
        help='Extracting caption information from images.'
    )
    parser.add_argument(
        '--image_path',
        default='../datasets/images/yfcc_images/',
        type=str,
        help='Images that require extraction of caption information.'
    )

    parser.add_argument(
        '--json_file',
        default='../datasets/yfcc_images_added_captions.json',
        type=str,
        help='JSON file containing images for which caption information extraction is desired.'
    )

    parser.add_argument(
        '--path_image_added_caption',
        default='../datasets/yfcc_images_added_captions.json',
        type=str,
        help=' '
    )

    parser.add_argument(
        '--seed',
        default=428,
        type=int,
        help='seed'
    )

    parser.add_argument(
        '--max_length',
        default=16, # 20 -> 30
        type=int,
        help=' '
    )

    parser.add_argument(
        '--num_beams',
        default=5, # 4->5
        type=int,
        help=' '
    )

    parser.add_argument(
        '--num_return_sequences',
        default=3,
        type=int,
        help='seed'
    )

    parser.add_argument(
        '--min_length',
        default=5,
        type=int,
        help='seed'
    )

    parser.add_argument(
        '--batch_size',
        default=20,
        type=int,
        help='seed'
    )

    args = parser.parse_args()
    return args
class YfccImageDataset(Dataset):
    def __init__(self, image_dir: str, processor: Blip2Processor, device:str):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.processor = processor
        self.device = device
    def __len__(self):
        return len(self.images)

    def read_image(self, img_path):
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        return img

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])
        image = self.read_image(img_path)
        image = self.processor(image, return_tensors="pt").pixel_values[0].to(self.device, torch.float16)

        return image, self.images[item]

def main(args):
    max_length = args.max_length
    num_beams = args.num_beams
    num_return_sequences = args.num_return_sequences
    min_length = args.min_length
    batch_size = args.batch_size
    path_image_added_caption = args.path_image_added_caption

    gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "num_return_sequences":num_return_sequences, "min_length":min_length, "no_repeat_ngram_size":3}

    processor = Blip2Processor.from_pretrained("../pretrained_models/blip2-opt-2.7b-coco/", local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained("../pretrained_models/blip2-opt-2.7b-coco/",
                                                         local_files_only=True, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    yfcc_image_dataset = YfccImageDataset(args.image_path, processor, device)
    yfcc_image_dataloader = DataLoader(yfcc_image_dataset, batch_size=batch_size, shuffle=False)

    yfcc_hash_to_caption = OrderedDict()

    with torch.no_grad():
        for p_v, image_hashes in tqdm(yfcc_image_dataloader):
            out = model.generate(p_v, **gen_kwargs)
            preds = processor.batch_decode(out, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]

            for i in range(len(preds)//num_return_sequences):
                yfcc_hash_to_caption[image_hashes[i]] = preds[i*num_return_sequences:(i+1)*num_return_sequences]

    with open(path_image_added_caption, 'w', encoding='utf-8') as file:
        json.dump(yfcc_hash_to_caption, file, indent="\t")


if __name__ == "__main__":
    args = parse_args()
    main(args)

