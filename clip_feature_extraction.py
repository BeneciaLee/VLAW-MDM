import json
import os
import gc

import numpy as np
import torch
from einops import rearrange
from patchify import patchify
from torchsr.models import edsr
from torchvision.transforms.functional import to_pil_image, to_tensor

from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor

from torchvision.transforms import Resize, PILToTensor
from tqdm import tqdm

from PIL import Image

class ImageDatasetForUpscaling(Dataset):
    def __init__(self, image_hased):
        self.image_hased = image_hased
        self.resize_672 = Resize((672, 672))

    def __len__(self):
        return len(self.image_hased)

    def __getitem__(self, item):
        pil_to_tensor = PILToTensor()
        image = Image.open(os.path.join("../datasets/images/yfcc_images", self.image_hased[item])).convert("RGB")
        image_tensor = pil_to_tensor(image)  # (c, h, w)
        return self.resize_672(image_tensor)
def extract_patches(img):
    # |img| = (bsz, c, h, w)
    size = 224  # patch size
    stride = 224  # patch stride
    patches = img.unfold(2, size, stride).unfold(3, size, stride)
    # |patches| = (bsz, c, 7, 7, 32, 32)
    bsz, chanel, p_h, p_w, h, w = patches.size()
    patches = rearrange(patches, "b c ph pw h w -> b c (ph pw) h w")
    # |patches| = (bsz, 3, 49, 32, 32)
    patches = patches.transpose(1,2)
    # |patches| = (bsz, 49, 3, 32, 32)
    return patches


def resize_for_images(data, h, w):
    # (bsz, c, h*3, w* 3)
    resize = Resize((h, w))
    bsz, _, _, _ = data.size()
    data = resize(data)
    return data

def resize_for_patches(data, h, w):
    resize = Resize((h, w))
    bsz, patch, _, _, _ = data.size()
    data = rearrange(data, 'b p c h w -> (b p) c h w')
    data = resize(data)
    data = rearrange(data, '(b p) c h w -> b p c h w', b = bsz, p = patch)
    return data

def image_resize_and_patch(dataset, batch_size=16):
    device = "cuda"
    model = CLIPModel.from_pretrained("../pretrained_models/clip-vit-base-patch32/",
                                      local_files_only=True).to(device)
    model.eval()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    patches = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            # |data| = (bsz, 3, 672, 672)
            data = data.to(device)
            images_reszied_patched = extract_patches(data)
            # print(f"images_reszied_patched -> {images_reszied_patched.shape}")
            # |images_reszied_patched| = (bsz, p, 3, 224, 224)
            b, p, c, h, w = images_reszied_patched.size()
            images_reszied_patched = rearrange(images_reszied_patched, 'b p c h w -> (b p) c h w')
            images_reszied_patched = processor.preprocess(images_reszied_patched, return_tensors="pt")['pixel_values']
            images_reszied_patched = model.get_image_features(images_reszied_patched.to(device))
            images_reszied_patched = rearrange(images_reszied_patched, '(b p) d -> b p d', b=b)
            patches.append(images_reszied_patched.to("cpu"))
    return torch.vstack(patches)

def image_up_scaling(dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    upscaling_datasets = []

    device = "cuda"
    model = edsr(scale=3, pretrained=True).eval()
    model = model.to(device)

    with torch.no_grad():
        for upscaling_datas in tqdm(dataloader):
            # |upscaling_datas| = (bsz, c, h, w)
            upscaling_datas_ = model(upscaling_datas.to(device))
            # |upscaling_dataset| = (bsz, c, h*3, w* 3)
            upscaling_datas_ = resize_for_images(upscaling_datas_, 1026, 1026)
            # |patches| = (bsz, c, 1026, 1026)
            upscaling_datas_ = extract_patches(upscaling_datas_)
            # |patches| = (bsz, p, 3, 342, 342)
            upscaling_datas_ = resize_for_patches(upscaling_datas_, 224, 224)
            # |patches| = (bsz, p, 3, 224, 224)
            upscaling_datasets.append(upscaling_datas_.cpu())
    return torch.vstack(upscaling_datasets)

def make_turn_one_data():
    train_path = '../datasets/train_valid_deleted_candidates.json'
    caption_json_file = '../datasets/yfcc_images_added_captions.json'
    with open(train_path) as file:
        datas = json.load(file)

    with open(caption_json_file) as file:
        caption_datas = json.load(file)

    captions = []
    image_hashes = []
    dialogues = []

    for data in tqdm(datas):
        dialog = data['dialog']
        image_hash = data['image_hash']
        tmp = []

        for d in dialog:
            tmp.append(" ".join(d))

        # And there are hashes that don't exist in the real data.
        # We don't need these hashes, so we remove them.
        if (image_hash + ".jpg") not in caption_datas or len(dialog) < 1:
            continue

        dialogues.append(tmp)
        captions.append(caption_datas[image_hash + ".jpg"][0])
        image_hashes.append(image_hash + ".jpg")
    print("###############################")
    # Captions: ['a view of a road with a stop sign and a building in the background']
    # Image_hashes: ['5eaa7034d31688ef1f9bed67f1f04f49.jpg']
    # Dialogues: [['Appreciative (Grateful)<sty>home sweet home', 'Glamorous<sty>in my big house', 'Appreciative (Grateful)<sty>Its a house, so like it']]
    print("Captions: ", captions[:1])
    print("Image_hashes : ", image_hashes[:1])
    print("Dialogues : ", dialogues[:1])

    del datas, caption_datas
    gc.collect()

    image_dataset_for_upscaling = ImageDatasetForUpscaling(image_hashes)
    patches = image_resize_and_patch(image_dataset_for_upscaling, batch_size=64)
    print("Patches : ", patches.shape)
    torch.save(patches, "../datasets/tensor_pactched_with_turn_one.pt")
    print("Success")

def make_test_data():
    test_path = '../datasets/test_deleted_candidates.json'
    caption_json_file = '../datasets/yfcc_images_added_captions.json'
    with open(test_path) as file:
        datas = json.load(file)

    with open(caption_json_file) as file:
        caption_datas = json.load(file)

    captions = []
    image_hashes = []
    dialogues = []

    for data in tqdm(datas):
        dialog = data['dialog']
        image_hash = data['image_hash']
        tmp = []

        for d in dialog:
            tmp.append(" ".join(d))

        if (image_hash + ".jpg") not in caption_datas or len(dialog) != 3:
            continue

        dialogues.append(tmp)
        captions.append(caption_datas[image_hash + ".jpg"][0])
        image_hashes.append(image_hash + ".jpg")
    print("###############################")
    # Captions: ['a view of a road with a stop sign and a building in the background']
    # Image_hashes: ['5eaa7034d31688ef1f9bed67f1f04f49.jpg']
    # Dialogues: [['Appreciative (Grateful)<sty>home sweet home', 'Glamorous<sty>in my big house', 'Appreciative (Grateful)<sty>Its a house, so like it']]
    print("Captions: ", captions[:1])
    print("Image_hashes : ", image_hashes[:1])
    print("Dialogues : ", dialogues[:1])

    del datas, caption_datas
    gc.collect()

    image_dataset_for_upscaling = ImageDatasetForUpscaling(image_hashes)
    patches = image_resize_and_patch(image_dataset_for_upscaling, batch_size=64)
    print("Patches : ", patches.shape)
    torch.save(patches, "../datasets/tensor_pactched_test.pt")
    print("Success")

def main():
    train_path = '../datasets/train_valid_deleted_candidates.json'
    caption_json_file = '../datasets/yfcc_images_added_captions.json'
    with open(train_path) as file:
        datas = json.load(file)

    with open(caption_json_file) as file:
        caption_datas = json.load(file)

    captions = []
    image_hashes = []
    dialogues = []

    for data in tqdm(datas):
        dialog = data['dialog']
        image_hash = data['image_hash']

        tmp = []

        for d in dialog:
            tmp.append(" ".join(d))

        if (image_hash + ".jpg") not in caption_datas or len(dialog) != 3:
            continue

        dialogues.append(tmp)
        captions.append(caption_datas[image_hash + ".jpg"][0])
        image_hashes.append(image_hash + ".jpg")
    print("###############################")
    # Captions: ['a view of a road with a stop sign and a building in the background']
    # Image_hashes: ['5eaa7034d31688ef1f9bed67f1f04f49.jpg']
    # Dialogues: [['Appreciative (Grateful)<sty>home sweet home', 'Glamorous<sty>in my big house', 'Appreciative (Grateful)<sty>Its a house, so like it']]
    print("Captions: ", captions[:1])
    print("Image_hashes : ", image_hashes[:1])
    print("Dialogues : ", dialogues[:1])

    del datas, caption_datas
    gc.collect()

    image_dataset_for_upscaling = ImageDatasetForUpscaling(image_hashes)
    patches = image_resize_and_patch(image_dataset_for_upscaling, batch_size=64)
    print("Patches : ", patches.shape)
    torch.save(patches, "../datasets/tensor_pactched.pt")
    print("Success")

if __name__ == "__main__":
    # main()
    # make_turn_one_data()
    make_test_data()