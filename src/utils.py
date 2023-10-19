"""
    shuffle_caption_data 함수는 batch 내에서 다른 caption으로 바꾼다.
    전체 caption 데이터에서 15%의 Caption 데이터를 다른 Caption 데이터로 바꾼다.
"""
import os
import torch
import copy
import json
import gc
import collections

from tqdm import tqdm
from dataclasses import dataclass
from typing import Union, Optional, Tuple, List

from .model.dataclass import *
from .model.dataset import *

# Original caption, modified caption, modified indices.
def get_random_indices(index_length: int, count: Optional[int] = None) -> torch.LongTensor:
    """
        Returns shuffled random numbers up to the length of index_length.

        If count is not None, it returns a specific number of items. If it's None, it returns all.
    """
    shuffled_indices = torch.randperm(index_length, dtype=torch.long)

    if count is not None:
        selected_indices = shuffled_indices[:count]
        return selected_indices

    return shuffled_indices

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def maksed_region_modeling(
        images: torch.FloatTensor,
        tokenizer,
        prob: float = 0.15,
) -> Union[Tuple[torch.Tensor], MaksedRegionModelingOutput]:
    bsz, patch, hs = images.size()

    number_of_selection = int(patch * prob)

    if number_of_selection <= 0:
        raise ValueError("Increase your batch_size")

    all_availabel_indices = get_random_indices(patch)
    selected_indices = all_availabel_indices[:number_of_selection]
    unselected_indices = all_availabel_indices[number_of_selection:]

    changed_images = copy.deepcopy(images)
    changed_images[:, selected_indices] = 0

    labels = changed_images.new_zeros(size=(bsz, patch), dtype=torch.long) + tokenizer.convert_tokens_to_ids("<feat>")
    labels[:, selected_indices] = tokenizer.convert_tokens_to_ids("<zero>")
    return MaksedRegionModelingOutput(
        modified_images=changed_images,
        labels=labels,
    )

def image_swapping(
        images: torch.FloatTensor,
        prob: float = 0.5,
        return_dict: bool = True
) -> Union[Tuple[torch.Tensor], ImageSwappingOutput]:
    """

        Args:
            captions(`torch.LongTensor` of shape `(batch_size, patch, hs)`)
                This is a data that contains caption information.
            prob (`float`)
                This is a value used to set the probability.
    """
    
    bsz, patch, hs = images.size()
    number_of_selection = int(bsz * prob)

    if number_of_selection <= 0:
        raise ValueError("Increase your batch_size")

    all_availabel_indices = get_random_indices(bsz)
    selected_indices = all_availabel_indices[:number_of_selection]
    unselected_indices = all_availabel_indices[number_of_selection:]

    if (len(selected_indices) + len(unselected_indices)) != bsz:
        raise ValueError(
            "The sum of the selected indices and unselected indices should be equal to the total number of indices, but they are different.")

    changed_images= copy.deepcopy(images)

    if len(unselected_indices) < number_of_selection:
        raise ValueError("The number of unselected_indices must always be greater than or equal to number_of_selection.")

    changed_images[selected_indices] = images[unselected_indices[:number_of_selection]]

    # labels = torch.zeros(size=(bsz,), dtype = torch.long)
    # labels[unselected_indices] = 1
    labels = ["positive" for idx in range(bsz)]
    for i in selected_indices:
        labels[i] = "negative"

    return ImageSwappingOutput(
        modified_images=changed_images,
        unmodified_indices=unselected_indices,
        modified_indices=selected_indices,
        number_of_selection=number_of_selection,
        labels=labels
    )

def change_dialogue_data(
        dialogues: Union[torch.LongTensor, List[str]],
        styles: List[str],
        tokenizer,
        prob: float = 0.5,
        return_dict: bool = True
) -> Union[Tuple[torch.Tensor], ChangeDialogueDataOutput]:
    """

        Args:
            captions(`torch.LongTensor` of shape `(batch_size, sequence_length)`)
                This is a data that contains caption information.
            prob (`float`)
                This is a value used to set the probability.
    """
    bsz = len(dialogues)
    number_of_selection = int(bsz * prob)

    if number_of_selection <= 0:
        raise ValueError("Increase your batch_size")

    all_availabel_indices = get_random_indices(bsz)
    selected_indices = all_availabel_indices[:number_of_selection]
    unselected_indices = all_availabel_indices[number_of_selection:]

    if (len(selected_indices) + len(unselected_indices)) != bsz:
        raise ValueError(
            "The sum of the selected indices and unselected indices should be equal to the total number of indices, but they are different.")

    dialogues_concat = []

    for style, d_l in zip(styles, dialogues):
        tmp = ""
        for s, dialogue in zip(style, d_l):
            tmp += (s + " " + dialogue)
        dialogues_concat.append(tokenizer(tmp, add_special_tokens=False, return_tensors="pt").input_ids[0])

    changed_dialogues_concat = copy.deepcopy(dialogues_concat)

    for up, i in enumerate(selected_indices):
        changed_dialogues_concat[i] = dialogues_concat[unselected_indices[up]]

    labels = ["positive" for idx in range(bsz)]
    for i in selected_indices:
        labels[i] = "negative"

    return ChangeDialogueDataOutput(
        original_dialgoues = dialogues_concat,
        modified_dialogues=changed_dialogues_concat,
        unmodified_indices=unselected_indices,
        modified_indices=selected_indices,
        number_of_selection=number_of_selection,
        labels=labels
    )


def masked_language_modeling(
        dialogues: Union[torch.LongTensor, List[str]],
        styles: List[str],
        tokenizer,
        truncation_max_length,
        prob: float = 0.2,
):
    bsz = len(dialogues)
    inputs = []
    targets = []
    lengths = []

    for turns, style in zip(dialogues, styles):
        turns_tok = []
        turns_tok_masked = []

        for sentence, s in zip(turns, style):
            # The reason for this distinction is:
            # There are samples with only turn1 and there are samples that have up to turn3.
            # Therefore, the two samples should be differentiated and handled accordingly.
            sentence_tok = tokenizer(sentence, add_special_tokens=False, return_tensors="pt", max_length=truncation_max_length, truncation=True).input_ids
            style_tok = tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids
            #  |sentence_tok| = (1, src_len)
            src_len = sentence_tok.shape[1]

            number_of_selection = int(src_len * prob)
            all_availabel_indices = get_random_indices(src_len)
            selected_indices = all_availabel_indices[:number_of_selection]

            sentence_tok_masked = sentence_tok.clone()
            # |sentence_tok_masked| = (1, src_len)
            sentence_tok_masked[0, selected_indices] = tokenizer.mask_token_id

            sentence_tok =  torch.cat([style_tok, sentence_tok], dim = -1)
            sentence_tok_masked = torch.cat([style_tok, sentence_tok_masked], dim = -1)
            turns_tok.append(sentence_tok)
            turns_tok_masked.append(sentence_tok_masked)
        turns_tok = torch.cat(turns_tok, dim = -1)
        turns_tok_masked = torch.cat(turns_tok_masked, dim = -1)
        inputs.append(turns_tok_masked[0])
        targets.append(turns_tok[0])
        lengths.append(turns_tok.shape[1])

    return MaskedLanguageModelingOutput(
        dialogue_tokenized=targets,
        dialogue_tokenized_masked=inputs,
        lengths=lengths
    )


def generation_captions(
        captions: Union[torch.LongTensor, List[str]],
        truncation_max_length,
        tokenizer,
):
    bsz = len(captions)
    inputs = []
    inputs_maksed = []
    targets = []
    lengths = []

    for sentence in captions:
        # Captions: ['a view of a road with a stop sign and a building in the background</s>']
        sentence_tok = tokenizer(sentence, add_special_tokens=False, return_tensors="pt", max_length=truncation_max_length, truncation=True).input_ids
        src_len = sentence_tok.shape[1]

        sentence_tok_masked = sentence_tok.clone()
        sentence_tok_masked[0, :-1] = tokenizer.mask_token_id

        inputs.append(sentence_tok[0])

        inputs_maksed.append(sentence_tok_masked[0])

        lengths.append(src_len)

    return MaskedCaptionModelingOutput(
        caption_tokenized=inputs,
        caption_tokenized_masked=inputs_maksed,
        lengths=lengths
    )

def masked_caption_modeling(
        captions: Union[torch.LongTensor, List[str]],
        tokenizer,
        prob: float = 0.2,
):
    bsz = len(captions)
    inputs = []
    inputs_maksed = []
    targets = []
    lengths = []

    for sentence in captions:
        # Captions: ['a view of a road with a stop sign and a building in the background</s>']
        sentence_tok = tokenizer(sentence, add_special_tokens=False, return_tensors="pt").input_ids
        src_len = sentence_tok.shape[1]

        # 1. Split the sentence into token units
        # 2. Randomly select indexes in the sentence to change (the 'number_of_selection' varies depending on the length of the sentence)
        # We need to differentiate between the original sentence and the masked sentence.
        # The masked sentence goes in as 'input_ids' value,
        # The original sentence goes in as the 'target' value.

        number_of_selection = int(src_len * prob)
        all_availabel_indices = get_random_indices(src_len)
        selected_indices = all_availabel_indices[:number_of_selection]

        sentence_tok_masked = sentence_tok.clone()
        sentence_tok_masked[0, selected_indices] = tokenizer.mask_token_id

        inputs.append(sentence_tok[0])

        inputs_maksed.append(sentence_tok_masked[0])

        lengths.append(src_len)

    return MaskedCaptionModelingOutput(
        caption_tokenized = inputs,
        caption_tokenized_masked=inputs_maksed,
        lengths=lengths
    )


def change_caption_data(
        captions: Union[torch.LongTensor, List[str]], # ['<cap>a view of a road with a stop sign and a building in the background']
        prob: float = 0.5,
        return_dict: bool = True
) -> Union[Tuple[torch.Tensor], ChangeCaptionDataOutput]:
    """
        This function samples data for pre-training on caption data.
        It randomly selects a certain number of data points within the entire batch_size and replaces
        their caption information with other captions, based on a predetermined probability value.

        Args:
            captions(`torch.LongTensor` of shape `(batch_size, sequence_length)`)
                This is a data that contains caption information.
            prob (`float`)
                This is a value used to set the probability.
    """

    # captions: List[str]
    bsz = len(captions)
    number_of_selection = int(bsz * prob)

    all_availabel_indices = get_random_indices(bsz)
    selected_indices = all_availabel_indices[:number_of_selection]
    unselected_indices = all_availabel_indices[number_of_selection:]

    if (len(selected_indices) + len(unselected_indices)) != bsz:
        raise ValueError(
            "The sum of the selected indices and unselected indices should be equal to the total number of indices, but they are different.")

    changed_captions = copy.deepcopy(captions)

    if len(unselected_indices) < number_of_selection:
        raise ValueError("The number of unselected_indices must always be greater than or equal to number_of_selection.")

    for idx, selected_idx in enumerate(selected_indices):
        changed_captions[selected_idx.item()] = captions[unselected_indices[idx]]

    labels = torch.zeros(size=(bsz, ), dtype=torch.long)
    labels[unselected_indices] = 1

    return ChangeCaptionDataOutput(
        original_captions=captions, # [str, str, str, ... str]
        modified_captions=changed_captions, # [str, str, str, ... str]
        unmodified_indices=unselected_indices,
        modified_indices=selected_indices,
        number_of_selection=number_of_selection,
        labels = labels
    )

def change_order_dialogues(
        dialogues: List[List[str]],
        prob: float = 0.75,
        return_type: Optional[str] = None,
        return_dict: bool = True,
) -> Union[Tuple[torch.Tensor], ChangeOrderDialoguesOutput]:
    """
        Shuffling the order information within a single dialogue.


        Args:
            dialogues(`torch.LongTensor` of shape `(batch_size, number_of_turns, sequence_length)`)
                This is a data that contains caption information.
            prob (`float`)
                This is a value used to set the probability.
            return_dict(`bool`, *optional*)
                Whether or not to return a [`ChangeOrderDialoguesOutput`] instead of a plain tuple.

    """
    # [[T1, T2, T3], [T1, T2, T3], ..., [T1, T2, T3]]
    bsz = len(dialogues)
    number_of_selection = int(bsz * prob)
    number_of_turns = len(dialogues[0])

    if number_of_selection <= 0:
        raise ValueError("Increase your bsz.")

    all_availabel_indices = get_random_indices(bsz)  # bsz 개수 만큼 랜덤하게 셔플된 인덱스를 반환
    selected_indices = all_availabel_indices[:number_of_selection] # 순서 정보가 바뀌는 데이터 인덱스
    unselected_indices = all_availabel_indices[number_of_selection:] # 순서 정보가 바뀌지 않는 데이터 인덱스

    changed_order_dialogues = copy.deepcopy(dialogues)
    grounded_orders = []

    for i in range(bsz):
        """
        There are indexes where the order of the dialogues changes and indexes where it doesn't.
        
        For the changing indexes, the corresponding changed order should be provided.
        For the unchanged indexes, the original order should be provided.
        
        This is stored in 'grounded_orders'. In other words, the answer is saved in the corresponding variable.
        """

        if i in selected_indices:
            dialogue_orders = get_random_indices(number_of_turns)

            for up in range(number_of_turns):
                changed_order_dialogues[i][up] = dialogues[i][dialogue_orders[up]]
            grounded_orders += [dialogue_orders]
            continue

        grounded_orders += [torch.arange(number_of_turns, dtype=torch.long)]

    changed_order_dialogues_concat = []

    for d_l in changed_order_dialogues:
        tmp = ""
        for dialogue in d_l:
            tmp += dialogue
        changed_order_dialogues_concat.append(tmp)

    if return_type == "pt":
        grounded_orders = torch.vstack(grounded_orders)
        if grounded_orders.shape[0] != bsz:
            raise ValueError("The number of grounded_orders must be the same as the bsz.")

    # labels = torch.zeros(size=(bsz,), dtype=torch.long)
    # labels[unselected_indices] = 1

    return ChangeOrderDialoguesOutput(
        modified_dialogues=changed_order_dialogues_concat,
        unmodified_indices=unselected_indices,
        modified_indices=selected_indices,
        number_of_selection=number_of_selection,
        labels=grounded_orders,
    )

def compute_f1(a_gold, a_pred, tokenizer):
    """
        you can check this https://rajpurkar.github.io/SQuAD-explorer/
    """
    gold_toks = tokenizer(a_gold, add_special_tokens=False).input_ids
    pred_toks = tokenizer(a_pred, add_special_tokens=False).input_ids

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
