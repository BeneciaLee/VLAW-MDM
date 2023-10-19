# A-Framework-for-Vision-Language-Warm-up-Tasks-in-Multimodal-Dialogue-Models


## Dependencies
* python == 3.7.5
* torch ==1.13.1
* transformers == 4.28.1
* numpy == 1.21.6
* nltk == 3.8.1
* pandas == 1.3.5
* pytorch-lightning == 1.9.5
* pytorch-transforemrs == 1.2.0
* einops == 0.6.1

## Feature Extraction & Caption Generation
Before initiating the essential Warm-up Tasks and Finetuning, it is imperative to preprocess the Image-Chat data for image feature extraction and caption generation.

The Image-Chat dataset can be downloaded via  [Image-Chat](https://parl.ai/projects/image_chat/)

For this dataset, image features are extracted using  [CLIP](https://github.com/openai/CLIP), and captions are generated through [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).

`clip_feature_extraction.py` : Extracts features from images.

`captioning_blip2-opt-2.7b.py` : Generates captions from images.

`captioning_blip2-flan-t5-xl-coco.py` : Generates captions from images.


## Warm-up Tasks
You can perform Warm-up tasks using `pretrain_preprocessing.py`.

## Fintuning Model
After performing the learning through the Warm-up tasks, you can import the trained weights and proceed with Fine-tuning. 

Fine-tuning can be carried out using `training_ablation.py`.

## Reference
```
@article{DBLP:journals/corr/abs-1811-00945,
author    = {Kurt Shuster and
           Samuel Humeau and
           Antoine Bordes and
           Jason Weston},
title     = {Engaging Image Chat: Modeling Personality in Grounded Dialogue},
journal   = {CoRR},
volume    = {abs/1811.00945},
year      = {2018},
url       = {http://arxiv.org/abs/1811.00945},
archivePrefix = {arXiv},
eprint    = {1811.00945},
timestamp = {Thu, 22 Nov 2018 17:58:30 +0100},
biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1811-00945},
bibsource = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{li2023blip2,
      title={{BLIP-2:} Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models}, 
      author={Junnan Li and Dongxu Li and Silvio Savarese and Steven Hoi},
      year={2023},
      booktitle={ICML},
}
```
