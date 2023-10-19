import torch

from dataclasses import dataclass
from typing import Union, Optional, Tuple, List, Dict, Set

@dataclass
class TrainingInputData:
    ccd_caption_dialogues: torch.LongTensor
    ccd_caption_dialogues_labels: List[str]
    iso_caption_dialogues: torch.LongTensor
    image_features: Union[List[torch.FloatTensor], torch.FloatTensor]
    image_features_swapped: Union[List[torch.FloatTensor], torch.FloatTensor]
    image_features_swapped_labels: torch.LongTensor
    cdd_caption_dialogues: torch.LongTensor
    cdd_caption_dialogues_labels: torch.LongTensor
    cod_caption_dialogues: torch.LongTensor
    cod_caption_dialogues_labels: torch.LongTensor

@dataclass
class GenerativePreTrainingInputData:
    image_features: Union[List[torch.FloatTensor], torch.FloatTensor]
    caption_dialogues: torch.LongTensor
    decoder_input_ids: torch.LongTensor
    labels: Optional[torch.LongTensor]

@dataclass
class GenerativeIntputData:
    caption_dialogues: torch.LongTensor
    image_features: torch.FloatTensor
    shape: torch.Size

@dataclass
class FinetuningtrainingData:
    encoder_input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    decoder_input_ids: torch.LongTensor
    decoder_labels: torch.LongTensor
    decoder_attention_mask: torch.LongTensor
    image_feature: torch.FloatTensor

@dataclass
class ValidationData:
    encoder_input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    decoder_input_ids: torch.LongTensor
    decoder_labels: torch.LongTensor
    decoder_attention_mask: torch.LongTensor
    decoder_str_labels: List[str]
    image_feature: torch.FloatTensor

@dataclass
class TestData:
    encoder_input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    decoder_input_ids: torch.LongTensor
    decoder_labels: torch.LongTensor
    decoder_attention_mask: torch.LongTensor
    decoder_str_labels: List[str]
    image_feature: torch.FloatTensor
    test_turn: str
    image_hashes: List[str]


@dataclass
class PretrainingData:
    # Task 1(MLM)
    mlm_caption_dialogues : torch.LongTensor
    mlm_decoder_inputs : torch.LongTensor
    mlm_decoder_labels : torch.LongTensor
    mlm_attention_mask : torch.LongTensor
    mlm_decoder_attention_mask : torch.LongTensor
    # Task 2(MRM)
    mrm_caption_dialogues: torch.LongTensor
    mrm_decoder_inputs: torch.LongTensor
    mrm_decoder_labels: torch.LongTensor
    mrm_attention_mask: torch.LongTensor
    mrm_decoder_attention_mask: torch.LongTensor
    # Task 3(CID)
    cid_caption_dialogues: torch.LongTensor
    cid_decoder_inputs: torch.LongTensor
    cid_decoder_labels: torch.LongTensor
    cid_attention_mask: torch.LongTensor
    cid_decoder_attention_mask : torch.LongTensor
    # Task 4(GCD)
    gcd_caption_dialogues: torch.LongTensor
    gcd_decoder_inputs: torch.LongTensor
    gcd_decoder_labels: torch.LongTensor
    gcd_attention_mask: torch.LongTensor
    gcd_decoder_attention_mask: torch.LongTensor
    image_feature : torch.FloatTensor
    modified_image_feature : torch.FloatTensor
    mrm_image_feature : torch.FloatTensor

@dataclass
class PretrainingDataWithoutCaption:
    # Task 1(MLM)
    mlm_caption_dialogues : torch.LongTensor
    mlm_decoder_inputs : torch.LongTensor
    mlm_decoder_labels : torch.LongTensor
    mlm_attention_mask : torch.LongTensor
    mlm_decoder_attention_mask : torch.LongTensor
    # Task 2(MRM)
    mrm_caption_dialogues: torch.LongTensor
    mrm_decoder_inputs: torch.LongTensor
    mrm_decoder_labels: torch.LongTensor
    mrm_attention_mask: torch.LongTensor
    mrm_decoder_attention_mask: torch.LongTensor
    # Task 3(CID)
    cid_caption_dialogues: torch.LongTensor
    cid_decoder_inputs: torch.LongTensor
    cid_decoder_labels: torch.LongTensor
    cid_attention_mask: torch.LongTensor
    cid_decoder_attention_mask : torch.LongTensor
    image_feature : torch.FloatTensor
    modified_image_feature : torch.FloatTensor
    mrm_image_feature : torch.FloatTensor

@dataclass
class GenerativeTrainingInputData:
    image_features: Union[List[torch.FloatTensor], torch.FloatTensor]
    caption_dialogues: torch.LongTensor
    attention_mask: torch.Tensor
    decoder_input: torch.LongTensor
    decoder_attention_mask: torch.Tensor
    labels: Optional[torch.LongTensor]
    task: str # ['Turn1', 'Turn2', 'Turn3']

@dataclass
class MaskedCaptionModelingOutput:
    caption_tokenized: List[torch.LongTensor] # <cap>Caption
    caption_tokenized_masked: List[torch.LongTensor] # <cap>CaptionMasked
    lengths: List[int]

@dataclass
class MaskedLanguageModelingOutput:
    dialogue_tokenized: torch.LongTensor
    dialogue_tokenized_masked: torch.LongTensor
    lengths: List[int]

@dataclass
class EmbedMultiModalOutput:
    image_length: int
    inputs_embeds: torch.FloatTensor
    # attention_mask: torch.FloatTensor


@dataclass
class MaksedRegionModelingOutput:
    modified_images: torch.FloatTensor
    labels: torch.LongTensor


@dataclass
class ImageSwappingOutput:
    modified_images: torch.FloatTensor
    unmodified_indices: torch.LongTensor
    modified_indices: torch.LongTensor
    number_of_selection: int
    labels: torch.LongTensor

@dataclass
class ChangeCaptionDataOutput:
    original_captions: torch.LongTensor
    modified_captions: torch.LongTensor
    unmodified_indices: torch.LongTensor
    modified_indices: torch.LongTensor
    number_of_selection: int
    labels: torch.LongTensor

@dataclass
class ChangeDialogueDataOutput:
    original_dialgoues : List[str]
    modified_dialogues: List[str]
    unmodified_indices: torch.LongTensor
    modified_indices: torch.LongTensor
    number_of_selection: int
    labels: torch.LongTensor
    
@dataclass
class ChangeOrderDialoguesOutput:
    modified_dialogues: torch.LongTensor
    unmodified_indices: torch.LongTensor
    modified_indices: torch.LongTensor
    number_of_selection: int
    labels: Union[List[torch.LongTensor], torch.LongTensor]
