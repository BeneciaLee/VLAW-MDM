import torch
import os

import numpy as np

from torchvision.transforms import Resize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer

from PIL import Image

from src.model.dataclass import *
from src.utils import *

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

def maksed_region_modeling(
        images: torch.FloatTensor,
        tokenizer,
        prob: float = 0.15,
) -> Union[Tuple[torch.Tensor], MaksedRegionModelingOutput]:
    bsz, patch, hs = images.size()

    number_of_selection = int(patch * prob)

    if number_of_selection <= 0 and patch != 1:
        raise ValueError("The number of bsz is too small to generate caption information. Please increase the bsz.")
    elif patch == 1:
        patch = 1


    if patch != 1:
        all_availabel_indices = get_random_indices(patch)  # bsz 개수 만큼 랜덤하게 셔플된 인덱스를 반환
        selected_indices = all_availabel_indices[:number_of_selection]
        unselected_indices = all_availabel_indices[number_of_selection:]

        changed_images = copy.deepcopy(images)
        changed_images[:, selected_indices] = 0

        labels = changed_images.new_zeros(size=(bsz, patch), dtype=torch.long) + tokenizer.convert_tokens_to_ids("<feat>")
        labels[:, selected_indices] = tokenizer.convert_tokens_to_ids("<zero>")
    else:
        changed_images = copy.deepcopy(images)
        changed_images[:, 0] = 0

        labels = changed_images.new_zeros(size=(bsz, patch), dtype=torch.long) + tokenizer.convert_tokens_to_ids(
            "<feat>")
        labels[:, 0] = tokenizer.convert_tokens_to_ids("<zero>")
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
        raise ValueError("The number of bsz is too small to generate caption information. Increase the number of bsz")

    all_availabel_indices = get_random_indices(bsz)
    selected_indices = all_availabel_indices[:number_of_selection]
    unselected_indices = all_availabel_indices[number_of_selection:]

    if (len(selected_indices) + len(unselected_indices)) != bsz:
        raise ValueError(
            "The sum of the selected indices and unselected indices should be equal to the total number of indices, but they are different.")

    changed_images = copy.deepcopy(images)

    if len(unselected_indices) < number_of_selection:
        raise ValueError("The count of unselected_indices must always be greater than or equal to number_of_selection")

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

def masked_language_modeling(
        dialogues: Union[torch.LongTensor, List[str]],
        styles: List[str],
        tokenizer,
        truncation_max_length,
        prob: float = 0.2,
):
    # captions come in as a List.
    # It's processed internally and then returned.
    bsz = len(dialogues)
    inputs = []
    targets = []
    lengths = []

    for turns, style in zip(dialogues, styles):
        turns_tok = []
        turns_tok_masked = []

        for sentence, s in zip(turns, style):
            # The reason for this distinction is:
            # There are samples that only have turn1 and there are samples that have up to turn3.
            # Therefore, the two samples should be differentiated and handled accordingly.
            sentence_tok = tokenizer(sentence, add_special_tokens=False, return_tensors="pt", max_length=truncation_max_length, truncation=True).input_ids
            style_tok = tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids
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

class PreVLPDialogueDataset(Dataset):
    def __init__(self, config, captions: List[str], image_features: torch.FloatTensor, dialogues: List[List[str]]):
        """
        Input Example :
        Captions:  ['a view of a road with a stop sign and a building in the background']
        Image_hashes :  ['5eaa7034d31688ef1f9bed67f1f04f49.jpg']
        Dialogues :  [['Appreciative (Grateful)<sty>home sweet home', 'Glamorous<sty>in my big house', 'Appreciative (Grateful)<sty>Its a house, so like it']]

        :param captions:
        :param images:
        :param dialogues:
        :param tokenizer:
        """
        super().__init__()
        self.config = config
        self.captions = captions
        self.image_features = image_features
        self.dialogues = dialogues
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Captions:  ['a view of a road with a stop sign and a building in the background']
        # Image_hashes :  ['5eaa7034d31688ef1f9bed67f1f04f49.jpg']
        # Dialogues :  [['Appreciative (Grateful)<sty>home sweet home', 'Glamorous<sty>in my big house', 'Appreciative (Grateful)<sty>Its a house, so like it']]

        return self.captions[idx], self.image_features[idx], self.dialogues[idx]


class PreFinetuningDataset(Dataset):
    """
    The reason for the prefix 'Pre' is that the image comes pre-extracted in tensor form.
    Therefore, the Dataset doesn't need to perform additional operations on the image.
    """

    def __init__(self, config, captions: List[str], image_features: torch.FloatTensor, styles: List[str],
                 turns: List[str]):
        super().__init__()
        self.config = config
        self.captions = captions
        self.image_features = image_features
        # self.dialogues = dialogues # 대화 정보들이 저장된다.
        self.styles = styles
        self.turns = turns
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Captions:  ['a view of a road with a stop sign and a building in the background']
        # Image_hashes :  ['5eaa7034d31688ef1f9bed67f1f04f49.jpg']
        # Dialogues :  [['Appreciative (Grateful) home sweet home', 'Glamorous in my big house', 'Appreciative (Grateful) Its a house, so like it']]

        return self.captions[idx], self.image_features[idx], self.styles[idx], self.turns[idx], self.image_hashes[idx]


class PreGenerateDataForPretraining():
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, batch):
        """
            batch = [(caption, image_feautres, dialogues), ..., (caption, image_feautres, dialogues)]
        :param batch:
        :return:
        """
        captions = []
        image_features = []
        styles = []
        raw_turns = []

        for caption, image, style, raw_turn in batch:
            captions.append(caption + " </s>")
            image_features.append(
                image.unsqueeze(0))

            tmp = []
            for turn in style:
                tmp.append(turn + " <sty>")
            styles.append(tmp)
            tmp = []
            for turn in raw_turn:
                tmp.append(turn + " </s>")
            raw_turns.append(tmp)

        image_features = torch.vstack(image_features)

        # cid, mrm, cgd, mlma
        changed_image_data_output = image_swapping(image_features)
        masked_region_modeling_output = maksed_region_modeling(image_features, self.tokenizer)
        caption_generation_output = generation_captions(captions, self.config.truncation_max_length, self.tokenizer)
        masked_language_modeling_output = masked_language_modeling(raw_turns, styles, self.tokenizer,
                                                                   self.config.truncation_max_length)

        # Caption + Dialogue
        changed_image_data_input_ids = []
        masked_region_modeling_input_ids = []
        caption_generation_output_input_ids = []
        masked_language_modeling_input_ids = []

        """
        The entities below are essentially one-dimensional in shape.
        That is, captions and dialogues with sizes like (src_len, ) will emerge.
        The reason for this is that to perform pad_sequence below, the data must be received as a list in one-dimensional form, not two-dimensional.
        Ex)
        Data should be stored in the list as [(src_len_1, ), (src_len_2), ... ] so that it can be transformed into a tensor with added padding.
        """
        # eos_token = image_features.new_zeros(size=(1,), dtype=torch.long) + self.tokenizer.eos_token_id
        bos_token = image_features.new_zeros(size=(1,)) + self.tokenizer.bos_token_id
        # print("@@@ -> ", bos_token.shape)
        # print("@@@ -> ", caption_generation_output.caption_tokenized[0].shape)
        for i in range(len(batch)):
            changed_image_data_input_ids.append(torch.cat([
                bos_token.clone().long(),
                caption_generation_output.caption_tokenized[i],
                masked_language_modeling_output.dialogue_tokenized[i]
            ], dim=-1))

            masked_region_modeling_input_ids.append(torch.cat([
                bos_token.clone().long(),
                caption_generation_output.caption_tokenized[i],
                masked_language_modeling_output.dialogue_tokenized[i]
            ], dim=-1))

            caption_generation_output_input_ids.append(torch.cat([
                bos_token.clone().long(),
                caption_generation_output.caption_tokenized_masked[i],
                masked_language_modeling_output.dialogue_tokenized[i]
            ], dim=-1))

            masked_language_modeling_input_ids.append(torch.cat([
                bos_token.clone().long(),
                caption_generation_output.caption_tokenized[i],
                masked_language_modeling_output.dialogue_tokenized_masked[i]
            ], dim=-1))
        """
            Part used as input to the Encoder

            CID
            MRM
            CGD
            MLM
        """

        # |masked_caption_modeling_input_ids| = (bsz, src_len)
        changed_image_data_input_ids = pad_sequence(changed_image_data_input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        masked_region_modeling_input_ids = pad_sequence(masked_region_modeling_input_ids, batch_first=True,
                                                        padding_value=self.tokenizer.pad_token_id)
        caption_generation_output_input_ids = pad_sequence(caption_generation_output_input_ids, batch_first=True,
                                                           padding_value=self.tokenizer.pad_token_id)
        masked_language_modeling_input_ids = pad_sequence(masked_language_modeling_input_ids, batch_first=True,
                                                          padding_value=self.tokenizer.pad_token_id)

        """For MCM, CID, MLM, the attention mask used in the Encoder is the same."""
        attention_mask = changed_image_data_input_ids.new_ones(size=(changed_image_data_input_ids.size()))
        attention_mask[changed_image_data_input_ids == self.tokenizer.pad_token_id] = 0
        image_pad = attention_mask.new_ones(size=(changed_image_data_input_ids.shape[0], image_features.shape[1] + 2))
        attention_mask = torch.cat([image_pad, attention_mask], dim=-1)

        """Since CDD has the Dialogue itself switched with something else, the overall length can change. Therefore, a separate attention mask is implemented."""
        # attention_mask_cdd = changed_dialogue_data_input_ids.new_ones(size=(changed_dialogue_data_input_ids.size()))
        # attention_mask_cdd[changed_dialogue_data_input_ids == self.tokenizer.pad_token_id] = 0
        # attention_mask_cdd = torch.cat([image_pad.clone(), attention_mask_cdd], dim = -1)

        """
            Part used as input to the Decoder

            MLM
            MRM
            CID
            CGD
        """

        """MLM"""
        mlm_decoder_labels = pad_sequence(masked_language_modeling_output.dialogue_tokenized, batch_first=True,
                                          padding_value=self.tokenizer.pad_token_id)
        bos_token = mlm_decoder_labels.new_zeros(size=(mlm_decoder_labels.shape[0], 1)) + self.tokenizer.bos_token_id
        eos_token = mlm_decoder_labels.new_zeros(size=(mlm_decoder_labels.shape[0], 1)) + self.tokenizer.eos_token_id

        mlm_decoder_labels = torch.cat([bos_token, mlm_decoder_labels], dim=-1)

        mlm_decoder_inputs = mlm_decoder_labels.clone()
        mlm_decoder_inputs[:, 1:] = mlm_decoder_inputs[:, :-1].clone()
        mlm_decoder_inputs[:, 1] = self.config.mlm_token_id
        mlm_decoder_attention_mask = mlm_decoder_inputs.new_ones(size=(mlm_decoder_inputs.size()))
        mlm_decoder_attention_mask[mlm_decoder_inputs == self.tokenizer.pad_token_id] = 0
        mlm_decoder_labels[:, 0] = self.config.mlm_token_id

        """MRM"""
        mrm_decoder_labels = masked_region_modeling_output.labels
        mrm_decoder_labels = torch.cat([bos_token, mrm_decoder_labels, eos_token], dim=-1)

        mrm_decoder_inputs = mrm_decoder_labels.clone()
        mrm_decoder_inputs[:, 1:] = mrm_decoder_inputs[:, :-1].clone()
        mrm_decoder_inputs[:, 1] = self.config.mrm_token_id
        mrm_decoder_attention_mask = mrm_decoder_inputs.new_ones(size=(mrm_decoder_inputs.size()))
        mrm_decoder_labels[:, 0] = self.config.mrm_token_id

        """CID"""
        cid_decoder_labels = self.tokenizer(changed_image_data_output.labels, return_tensors="pt", padding=True)
        cid_decoder_attention_mask = cid_decoder_labels.attention_mask
        cid_decoder_labels = cid_decoder_labels.input_ids
        cid_decoder_inputs = cid_decoder_labels.clone()
        cid_decoder_inputs[:, 1:] = cid_decoder_inputs[:, :-1].clone()
        cid_decoder_inputs[:, 1] = self.config.cid_token_id
        cid_decoder_labels[:, 0] = self.config.cid_token_id

        """GCD"""
        gcd_decoder_labels = pad_sequence(caption_generation_output.caption_tokenized, batch_first=True,
                                          padding_value=self.tokenizer.pad_token_id)

        gcd_decoder_labels = torch.cat([bos_token, gcd_decoder_labels, eos_token], dim=-1)

        gcd_decoder_inputs = gcd_decoder_labels.clone()
        gcd_decoder_inputs[:, 1:] = gcd_decoder_inputs[:, :-1].clone()
        gcd_decoder_inputs[:, 1] = self.config.gcd_token_id
        gcd_decoder_attention_mask = gcd_decoder_inputs.new_ones(size=(gcd_decoder_inputs.size()))
        gcd_decoder_attention_mask[gcd_decoder_inputs == self.tokenizer.pad_token_id] = 0
        gcd_decoder_labels[:, 0] = self.config.gcd_token_id

        return PretrainingData(
            # Task 1(MLM)
            mlm_caption_dialogues=masked_language_modeling_input_ids,
            mlm_decoder_inputs=mlm_decoder_inputs,
            mlm_decoder_labels=mlm_decoder_labels,
            mlm_attention_mask=attention_mask,
            mlm_decoder_attention_mask=mlm_decoder_attention_mask,
            image_feature=image_features,
            # Task 2(MRM)
            mrm_caption_dialogues=masked_region_modeling_input_ids,
            mrm_decoder_inputs=mrm_decoder_inputs,
            mrm_decoder_labels=mrm_decoder_labels,
            mrm_attention_mask=attention_mask,
            mrm_decoder_attention_mask=mrm_decoder_attention_mask,
            # Task 3(CID)
            cid_caption_dialogues=changed_image_data_input_ids,
            cid_decoder_inputs=cid_decoder_inputs,
            cid_decoder_labels=cid_decoder_labels,
            cid_attention_mask=attention_mask,
            cid_decoder_attention_mask=cid_decoder_attention_mask,
            # Task 4(GCD)
            gcd_caption_dialogues=caption_generation_output_input_ids,
            gcd_decoder_inputs=gcd_decoder_inputs,
            gcd_decoder_labels=gcd_decoder_labels,
            gcd_attention_mask=attention_mask,
            gcd_decoder_attention_mask=gcd_decoder_attention_mask,
            modified_image_feature=changed_image_data_output.modified_images,
            mrm_image_feature=masked_region_modeling_output.modified_images
        )
# PreGenerateDataForTestWithoutCaption
class PreGenerateDataForPretrainingWithoutCaption():
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, batch):
        captions = []
        image_features = []
        styles = []
        raw_turns = []

        for caption, image, style, raw_turn in batch:
            captions.append(caption + " </s>")
            image_features.append(
                image.unsqueeze(0))  # [torch.FloatTensor: (1, 50, 512), torch.FloatTensor: (1, 50, 512), ... ]

            tmp = []
            for turn in style:
                tmp.append(turn + " <sty>")
            styles.append(tmp)
            tmp = []
            for turn in raw_turn:
                tmp.append(turn + " </s>")
            raw_turns.append(tmp)

        image_features = torch.vstack(image_features)

        # cid, mrm, cgd, mlma
        changed_image_data_output = image_swapping(image_features)
        masked_region_modeling_output = maksed_region_modeling(image_features, self.tokenizer)
        masked_language_modeling_output = masked_language_modeling(raw_turns, styles, self.tokenizer,
                                                                   self.config.truncation_max_length)

        # Caption + Dialogue
        changed_image_data_input_ids = []
        masked_region_modeling_input_ids = []
        masked_language_modeling_input_ids = []

        bos_token = image_features.new_zeros(size=(1,)) + self.tokenizer.bos_token_id
        for i in range(len(batch)):
            changed_image_data_input_ids.append(torch.cat([
                bos_token.clone().long(),
                masked_language_modeling_output.dialogue_tokenized[i]
            ], dim=-1))

            masked_region_modeling_input_ids.append(torch.cat([
                bos_token.clone().long(),
                masked_language_modeling_output.dialogue_tokenized[i]
            ], dim=-1))

            masked_language_modeling_input_ids.append(torch.cat([
                bos_token.clone().long(),
                masked_language_modeling_output.dialogue_tokenized_masked[i]
            ], dim=-1))

        changed_image_data_input_ids = pad_sequence(changed_image_data_input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        masked_region_modeling_input_ids = pad_sequence(masked_region_modeling_input_ids, batch_first=True,
                                                        padding_value=self.tokenizer.pad_token_id)
        masked_language_modeling_input_ids = pad_sequence(masked_language_modeling_input_ids, batch_first=True,
                                                          padding_value=self.tokenizer.pad_token_id)

        attention_mask = changed_image_data_input_ids.new_ones(size=(changed_image_data_input_ids.size()))
        attention_mask[changed_image_data_input_ids == self.tokenizer.pad_token_id] = 0
        image_pad = attention_mask.new_ones(size=(changed_image_data_input_ids.shape[0], image_features.shape[1] + 2))
        attention_mask = torch.cat([image_pad, attention_mask], dim=-1)

        mlm_decoder_labels = pad_sequence(masked_language_modeling_output.dialogue_tokenized, batch_first=True,
                                          padding_value=self.tokenizer.pad_token_id)
        bos_token = mlm_decoder_labels.new_zeros(size=(mlm_decoder_labels.shape[0], 1)) + self.tokenizer.bos_token_id
        eos_token = mlm_decoder_labels.new_zeros(size=(mlm_decoder_labels.shape[0], 1)) + self.tokenizer.eos_token_id

        mlm_decoder_labels = torch.cat([bos_token, mlm_decoder_labels], dim=-1)

        mlm_decoder_inputs = mlm_decoder_labels.clone()
        mlm_decoder_inputs[:, 1:] = mlm_decoder_inputs[:, :-1].clone()
        mlm_decoder_inputs[:, 1] = self.config.mlm_token_id
        mlm_decoder_attention_mask = mlm_decoder_inputs.new_ones(size=(mlm_decoder_inputs.size()))
        mlm_decoder_attention_mask[mlm_decoder_inputs == self.tokenizer.pad_token_id] = 0
        mlm_decoder_labels[:, 0] = self.config.mlm_token_id

        """MRM"""
        mrm_decoder_labels = masked_region_modeling_output.labels
        mrm_decoder_labels = torch.cat([bos_token, mrm_decoder_labels, eos_token], dim=-1)

        mrm_decoder_inputs = mrm_decoder_labels.clone()
        mrm_decoder_inputs[:, 1:] = mrm_decoder_inputs[:, :-1].clone()
        mrm_decoder_inputs[:, 1] = self.config.mrm_token_id
        mrm_decoder_attention_mask = mrm_decoder_inputs.new_ones(size=(mrm_decoder_inputs.size()))
        mrm_decoder_labels[:, 0] = self.config.mrm_token_id

        """CID"""
        cid_decoder_labels = self.tokenizer(changed_image_data_output.labels, return_tensors="pt", padding=True)
        cid_decoder_attention_mask = cid_decoder_labels.attention_mask
        cid_decoder_labels = cid_decoder_labels.input_ids
        cid_decoder_inputs = cid_decoder_labels.clone()
        cid_decoder_inputs[:, 1:] = cid_decoder_inputs[:, :-1].clone()
        cid_decoder_inputs[:, 1] = self.config.cid_token_id
        cid_decoder_labels[:, 0] = self.config.cid_token_id

        return PretrainingDataWithoutCaption(
            # Task 1(MLM)
            mlm_caption_dialogues=masked_language_modeling_input_ids,
            mlm_decoder_inputs=mlm_decoder_inputs,
            mlm_decoder_labels=mlm_decoder_labels,
            mlm_attention_mask=attention_mask,
            mlm_decoder_attention_mask=mlm_decoder_attention_mask,
            image_feature=image_features,
            # Task 2(MRM)
            mrm_caption_dialogues=masked_region_modeling_input_ids,
            mrm_decoder_inputs=mrm_decoder_inputs,
            mrm_decoder_labels=mrm_decoder_labels,
            mrm_attention_mask=attention_mask,
            mrm_decoder_attention_mask=mrm_decoder_attention_mask,
            # Task 3(CID)
            cid_caption_dialogues=changed_image_data_input_ids,
            cid_decoder_inputs=cid_decoder_inputs,
            cid_decoder_labels=cid_decoder_labels,
            cid_attention_mask=attention_mask,
            cid_decoder_attention_mask=cid_decoder_attention_mask,
            modified_image_feature=changed_image_data_output.modified_images,
            mrm_image_feature=masked_region_modeling_output.modified_images
        )

class PreGenerateDataForFinetuning():

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config

    def truncaction_str_to_str(self, sentence, max_length=40):
        return self.tokenizer.batch_decode(self.tokenizer(sentence,
                                                          return_tensors='pt',
                                                          add_special_tokens=False,
                                                          max_length=max_length,
                                                          truncation=True).input_ids)[0].strip()

    def __call__(self, batch):
        captions = []
        image_features = []
        styles = []
        raw_turns = []

        for caption, image, style, raw_turn in batch:
            captions.append(caption + " </s>")
            image_features.append(
                image.unsqueeze(0))  # [torch.FloatTensor: (1, 50, 512), torch.FloatTensor: (1, 50, 512), ... ]

            tmp = []
            for turn in style:
                tmp.append(turn + " <sty>")
            styles.append(tmp)
            tmp = []
            for turn in raw_turn:
                tmp.append(turn + " </s>")
            raw_turns.append(tmp)

        image_features = torch.vstack(image_features)

        encoder_caption_dialogue = []
        decoder_caption_dialogue = []

        for i in range(len(batch)):
            caption = captions[i]
            if len(raw_turns[i]) == 3:
                coin = torch.randint(0, 3, size=(1,))
                if coin == 0:
                    turn = raw_turns[i][0]
                    style = styles[i][0]
                    encoder_caption_dialogue.append(caption + style + "</s>")
                    decoder_caption_dialogue.append(self.truncaction_str_to_str(turn, max_length=253))
                elif coin == 1:
                    turn_one = raw_turns[i][0]
                    turn_two = raw_turns[i][1]

                    style_one = styles[i][0]
                    style_two = styles[i][1]

                    encoder_caption_dialogue.append(
                        caption + style_one + self.truncaction_str_to_str(turn_one) + style_one + "</s>")
                    decoder_caption_dialogue.append(self.truncaction_str_to_str(turn_two, max_length=253))
                elif coin == 2:
                    turn_one = raw_turns[i][0]
                    turn_two = raw_turns[i][1]
                    turn_three = raw_turns[i][2]

                    style_one = styles[i][0]
                    style_two = styles[i][1]
                    style_three = styles[i][2]

                    encoder_caption_dialogue.append(caption + style_one + self.truncaction_str_to_str(
                        turn_one) + style_two + self.truncaction_str_to_str(turn_two) + style_three + "</s>")
                    decoder_caption_dialogue.append(self.truncaction_str_to_str(turn_three, max_length=253))
            else:
                turn = raw_turns[i][0]
                style = styles[i][0]

                # <sty>style1
                encoder_caption_dialogue.append(caption + style + "</s>")
                decoder_caption_dialogue.append(self.truncaction_str_to_str(turn, max_length=253))

        """
            위에서 보면 알지만 encoder_caption_dialogue는 tokenizer에서 special_token을 False로 해줘야함.
            반대로 decoder_caption_dialogue는 tokenizer에서 special_token을 True로 해줘야함.
        """
        encoder_input_ids = self.tokenizer(encoder_caption_dialogue, add_special_tokens=False, padding=True,
                                           return_tensors="pt", max_length=self.config.max_position_embeddings,
                                           truncation=True)

        encoder_attention_mask = encoder_input_ids.attention_mask
        image_pad = encoder_attention_mask.new_ones(size=(encoder_attention_mask.shape[0], image_features.shape[1] + 2))
        encoder_attention_mask = torch.cat([image_pad, encoder_attention_mask], dim=-1)

        encoder_input_ids = encoder_input_ids.input_ids

        decoder_labels = self.tokenizer(decoder_caption_dialogue, add_special_tokens=True, padding=True,
                                        return_tensors="pt", max_length=(self.config.max_position_embeddings - 2),
                                        truncation=True)

        # <s><sty>style<utr>utterance</s>
        decoder_attention_mask = decoder_labels.attention_mask
        decoder_labels = decoder_labels.input_ids

        # <s><s><sty>style<utr>utterance
        decoder_input_ids = shift_tokens_right(decoder_labels, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id)
        decoder_attention_mask[:, 1:] = decoder_attention_mask[:, :-1].clone()
        decoder_attention_mask[:, 0] = 1

        return FinetuningtrainingData(
            encoder_input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_labels=decoder_labels,
            image_feature=image_features
        )

class PreGenerateDataForFinetuningWithoutCaption():
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config

    def truncaction_str_to_str(self, sentence, max_length=40):
        return self.tokenizer.batch_decode(self.tokenizer(sentence,
                                                          return_tensors='pt',
                                                          add_special_tokens=False,
                                                          max_length=max_length,
                                                          truncation=True).input_ids)[0].strip()

    def __call__(self, batch):
        captions = []
        image_features = []
        styles = []
        raw_turns = []

        for caption, image, style, raw_turn in batch:
            captions.append(caption + " </s>")
            image_features.append(
                image.unsqueeze(0))  # [torch.FloatTensor: (1, 50, 512), torch.FloatTensor: (1, 50, 512), ... ]

            tmp = []
            for turn in style:
                tmp.append(turn + " <sty>")
            styles.append(tmp)
            tmp = []
            for turn in raw_turn:
                tmp.append(turn + " </s>")
            raw_turns.append(tmp)

        image_features = torch.vstack(image_features)

        encoder_caption_dialogue = []
        decoder_caption_dialogue = []

        for i in range(len(batch)):
            if len(raw_turns[i]) == 3:
                coin = torch.randint(0, 3, size=(1,))
                if coin == 0:
                    turn = raw_turns[i][0]
                    style = styles[i][0]
                    encoder_caption_dialogue.append(style + "</s>")
                    decoder_caption_dialogue.append(self.truncaction_str_to_str(turn, max_length=253))
                elif coin == 1:
                    turn_one = raw_turns[i][0]
                    turn_two = raw_turns[i][1]

                    style_one = styles[i][0]
                    style_two = styles[i][1]

                    encoder_caption_dialogue.append(
                        style_one + self.truncaction_str_to_str(turn_one) + style_one + "</s>")
                    decoder_caption_dialogue.append(self.truncaction_str_to_str(turn_two, max_length=253))
                elif coin == 2:
                    turn_one = raw_turns[i][0]
                    turn_two = raw_turns[i][1]
                    turn_three = raw_turns[i][2]

                    style_one = styles[i][0]
                    style_two = styles[i][1]
                    style_three = styles[i][2]

                    encoder_caption_dialogue.append(style_one + self.truncaction_str_to_str(
                        turn_one) + style_two + self.truncaction_str_to_str(turn_two) + style_three + "</s>")
                    decoder_caption_dialogue.append(self.truncaction_str_to_str(turn_three, max_length=253))
            else:
                turn = raw_turns[i][0]
                style = styles[i][0]

                encoder_caption_dialogue.append(style + "</s>")
                decoder_caption_dialogue.append(self.truncaction_str_to_str(turn, max_length=253))

        encoder_input_ids = self.tokenizer(encoder_caption_dialogue, add_special_tokens=False, padding=True,
                                           return_tensors="pt", max_length=self.config.max_position_embeddings,
                                           truncation=True)

        encoder_attention_mask = encoder_input_ids.attention_mask
        image_pad = encoder_attention_mask.new_ones(size=(encoder_attention_mask.shape[0], image_features.shape[1] + 2))
        encoder_attention_mask = torch.cat([image_pad, encoder_attention_mask], dim=-1)

        encoder_input_ids = encoder_input_ids.input_ids

        decoder_labels = self.tokenizer(decoder_caption_dialogue, add_special_tokens=True, padding=True,
                                        return_tensors="pt", max_length=(self.config.max_position_embeddings - 2),
                                        truncation=True)

        decoder_attention_mask = decoder_labels.attention_mask
        decoder_labels = decoder_labels.input_ids

        # <s><s><sty>style<utr>utterance
        decoder_input_ids = shift_tokens_right(decoder_labels, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id)
        decoder_attention_mask[:, 1:] = decoder_attention_mask[:, :-1].clone()
        decoder_attention_mask[:, 0] = 1

        return FinetuningtrainingData(
            encoder_input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_labels=decoder_labels,
            image_feature=image_features
        )
class PreGenerateDataForValidation():

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config

    def truncaction_str_to_str(self, sentence, max_length=40):
        return self.tokenizer.batch_decode(self.tokenizer(sentence,
                                                          return_tensors='pt',
                                                          add_special_tokens=False,
                                                          max_length=max_length,
                                                          truncation=True).input_ids)[0].strip()

    def __call__(self, batch):
        captions = []
        image_features = []
        styles = []
        raw_turns = []
        for caption, image, style, raw_turn in batch:
            captions.append(caption + "</s>")
            image_features.append(
                image.unsqueeze(0))  # [torch.FloatTensor: (1, 50, 512), torch.FloatTensor: (1, 50, 512), ... ]

            tmp = []
            for turn in style:
                tmp.append(turn + "<sty>")
            styles.append(tmp)
            tmp = []
            for turn in raw_turn:
                tmp.append(turn + "</s>")
            raw_turns.append(tmp)

        image_features = torch.vstack(image_features)

        encoder_caption_dialogue = []
        decoder_caption_dialogue = []

        for i in range(len(batch)):
            caption = captions[i]
            if len(raw_turns[i]) == 3:
                coin = torch.randint(0, 3, size=(1,))

                if coin == 0:
                    turn = raw_turns[i][0]
                    style = styles[i][0]
                    encoder_caption_dialogue.append(caption + style + "</s>")
                    decoder_caption_dialogue.append(turn)
                elif coin == 1:
                    turn_one = raw_turns[i][0]
                    turn_two = raw_turns[i][1]

                    style_one = styles[i][0]
                    style_two = styles[i][1]

                    encoder_caption_dialogue.append(
                        caption + style_one + self.truncaction_str_to_str(turn_one) + style_two + "</s>")
                    decoder_caption_dialogue.append(turn_two)
                elif coin == 2:
                    turn_one = raw_turns[i][0]
                    turn_two = raw_turns[i][1]
                    turn_three = raw_turns[i][2]

                    style_one = styles[i][0]
                    style_two = styles[i][1]
                    style_three = styles[i][2]

                    encoder_caption_dialogue.append(caption + style_one + self.truncaction_str_to_str(
                        turn_one) + style_two + self.truncaction_str_to_str(turn_two) + style_three + "</s>")
                    decoder_caption_dialogue.append(turn_three)
            else:
                turn = raw_turns[i][0]
                style = styles[i][0]
                encoder_caption_dialogue.append(caption + style + "</s>")
                decoder_caption_dialogue.append(turn)

        encoder_input_ids = self.tokenizer(encoder_caption_dialogue, add_special_tokens=False, padding=True,
                                           return_tensors="pt", max_length=self.config.max_position_embeddings,
                                           truncation=True)

        encoder_attention_mask = encoder_input_ids.attention_mask
        image_pad = encoder_attention_mask.new_ones(size=(encoder_attention_mask.shape[0], image_features.shape[1] + 2))
        encoder_attention_mask = torch.cat([image_pad, encoder_attention_mask], dim=-1)

        encoder_input_ids = encoder_input_ids.input_ids

        decoder_labels = self.tokenizer(decoder_caption_dialogue, add_special_tokens=False, padding=True,
                                        return_tensors="pt", max_length=(self.config.max_position_embeddings - 2),
                                        truncation=True)

        decoder_attention_mask = decoder_labels.attention_mask
        decoder_labels = decoder_labels.input_ids

        decoder_input_ids = shift_tokens_right(decoder_labels, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id)
        decoder_attention_mask[:, 1:] = decoder_attention_mask[:, :-1].clone()
        decoder_attention_mask[:, 0] = 1

        return ValidationData(
            encoder_input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_labels=decoder_labels,
            decoder_str_labels=decoder_caption_dialogue,
            image_feature=image_features
        )


class PreGenerateDataForTest():

    def __init__(self, config, tokenizer, test_turn = None):
        self.tokenizer = tokenizer
        self.config = config
        self.test_turn = test_turn

    def truncaction_str_to_str(self, sentence, max_length=40):
        return self.tokenizer.batch_decode(self.tokenizer(sentence,
                                                          return_tensors='pt',
                                                          add_special_tokens=False,
                                                          max_length=max_length,
                                                          truncation=True).input_ids)[0].strip()

    def __call__(self, batch):
        captions = []
        image_features = []
        styles = []
        raw_turns = []
        image_hashes = []

        for caption, image, style, raw_turn, image_hash in batch:
            captions.append(caption + "</s>")
            image_features.append(
                image.unsqueeze(0))

            image_hashes.append(image_hash)

            tmp = []
            for turn in style:
                tmp.append(turn + "<sty>")
            styles.append(tmp)
            tmp = []
            for turn in raw_turn:
                tmp.append(turn + "</s>")
            raw_turns.append(tmp)

        image_features = torch.vstack(image_features)

        encoder_caption_dialogue = []
        decoder_caption_dialogue = []

        mapping_test_turn = {
            'turn_one': 0,
            'turn_two': 1,
            'turn_three': 2
        }
        selected_image_index = []

        for i in range(len(batch)):
            caption = captions[i]
            if len(raw_turns[i]) == 3:
                selected_image_index.append(i)
                coin = mapping_test_turn[self.test_turn]

                if coin == 0:
                    turn = raw_turns[i][0]
                    style = styles[i][0]
                    encoder_caption_dialogue.append(caption + style + "</s>")
                    decoder_caption_dialogue.append(turn)
                elif coin == 1:
                    turn_one = raw_turns[i][0]
                    turn_two = raw_turns[i][1]

                    style_one = styles[i][0]
                    style_two = styles[i][1]

                    encoder_caption_dialogue.append(
                        caption + style_one + self.truncaction_str_to_str(turn_one) + style_two + "</s>")
                    decoder_caption_dialogue.append(turn_two)
                elif coin == 2:
                    turn_one = raw_turns[i][0]
                    turn_two = raw_turns[i][1]
                    turn_three = raw_turns[i][2]

                    style_one = styles[i][0]
                    style_two = styles[i][1]
                    style_three = styles[i][2]

                    encoder_caption_dialogue.append(caption + style_one + self.truncaction_str_to_str(
                        turn_one) + style_two + self.truncaction_str_to_str(turn_two) + style_three + "</s>")
                    decoder_caption_dialogue.append(turn_three)
            else:
                if self.config.test_turn == "turn_one":
                    selected_image_index.append(i)
                    turn = raw_turns[i][0]
                    style = styles[i][0]

                    encoder_caption_dialogue.append(caption + style + "</s>")
                    decoder_caption_dialogue.append(turn)

        image_features = image_features[selected_image_index]
        encoder_input_ids = self.tokenizer(encoder_caption_dialogue, add_special_tokens=False, padding=True,
                                           return_tensors="pt")

        encoder_attention_mask = encoder_input_ids.attention_mask
        image_pad = encoder_attention_mask.new_ones(size=(encoder_attention_mask.shape[0], image_features.shape[1] + 2))
        encoder_attention_mask = torch.cat([image_pad, encoder_attention_mask], dim=-1)

        encoder_input_ids = encoder_input_ids.input_ids

        # decoder_labels = self.tokenizer(decoder_caption_dialogue, add_special_tokens=True, padding=True,
        #                                 return_tensors="pt")
        decoder_labels = self.tokenizer(decoder_caption_dialogue, add_special_tokens=False, padding=True,
                                        return_tensors="pt")

        decoder_attention_mask = decoder_labels.attention_mask
        decoder_labels = decoder_labels.input_ids

        # <s><s><sty>style<utr>utterance
        decoder_input_ids = shift_tokens_right(decoder_labels, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id)
        decoder_attention_mask[:, 1:] = decoder_attention_mask[:, :-1].clone()
        decoder_attention_mask[:, 0] = 1

        return TestData(
            encoder_input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_labels=decoder_labels,
            decoder_str_labels=decoder_caption_dialogue,
            image_feature=image_features,
            test_turn=self.test_turn,
            image_hashes=image_hashes
        )

class PreGenerateDataForTestWithoutCaption():
    def __init__(self, config, tokenizer, test_turn = None):
        self.tokenizer = tokenizer
        self.config = config
        self.test_turn = test_turn

    def truncaction_str_to_str(self, sentence, max_length=40):
        return self.tokenizer.batch_decode(self.tokenizer(sentence,
                                                          return_tensors='pt',
                                                          add_special_tokens=False,
                                                          max_length=max_length,
                                                          truncation=True).input_ids)[0].strip()

    def __call__(self, batch):
        captions = []
        image_features = []
        styles = []
        raw_turns = []

        for caption, image, style, raw_turn in batch:
            captions.append(caption + "</s>")  # caption 정보 앞에 <cap>이라는 특수 토큰을 추가한다.
            image_features.append(
                image.unsqueeze(0))  # [torch.FloatTensor: (1, 50, 512), torch.FloatTensor: (1, 50, 512), ... ]

            tmp = []
            for turn in style:
                tmp.append(turn + "<sty>")
            styles.append(tmp)
            tmp = []
            for turn in raw_turn:
                tmp.append(turn + "</s>")
            raw_turns.append(tmp)

        image_features = torch.vstack(image_features)

        encoder_caption_dialogue = []
        decoder_caption_dialogue = []

        mapping_test_turn = {
            'turn_one': 0,
            'turn_two': 1,
            'turn_three': 2
        }
        selected_image_index = []

        for i in range(len(batch)):
            if len(raw_turns[i]) == 3:
                selected_image_index.append(i)
                coin = mapping_test_turn[self.test_turn]

                if coin == 0:
                    turn = raw_turns[i][0]
                    style = styles[i][0]
                    encoder_caption_dialogue.append(style + "</s>")
                    decoder_caption_dialogue.append(turn)
                elif coin == 1:
                    turn_one = raw_turns[i][0]
                    turn_two = raw_turns[i][1]

                    style_one = styles[i][0]
                    style_two = styles[i][1]

                    encoder_caption_dialogue.append(
                        style_one + self.truncaction_str_to_str(turn_one) + style_two + "</s>")
                    decoder_caption_dialogue.append(turn_two)
                elif coin == 2:
                    turn_one = raw_turns[i][0]
                    turn_two = raw_turns[i][1]
                    turn_three = raw_turns[i][2]

                    style_one = styles[i][0]
                    style_two = styles[i][1]
                    style_three = styles[i][2]

                    encoder_caption_dialogue.append(style_one + self.truncaction_str_to_str(
                        turn_one) + style_two + self.truncaction_str_to_str(turn_two) + style_three + "</s>")
                    decoder_caption_dialogue.append(turn_three)
            else:
                if self.config.test_turn == "turn_one":
                    selected_image_index.append(i)
                    turn = raw_turns[i][0]
                    style = styles[i][0]

                    encoder_caption_dialogue.append(style + "</s>")
                    decoder_caption_dialogue.append(turn)

        image_features = image_features[selected_image_index]
        encoder_input_ids = self.tokenizer(encoder_caption_dialogue, add_special_tokens=False, padding=True,
                                           return_tensors="pt")

        encoder_attention_mask = encoder_input_ids.attention_mask
        image_pad = encoder_attention_mask.new_ones(size=(encoder_attention_mask.shape[0], image_features.shape[1] + 2))
        encoder_attention_mask = torch.cat([image_pad, encoder_attention_mask], dim=-1)

        encoder_input_ids = encoder_input_ids.input_ids

        # decoder_labels = self.tokenizer(decoder_caption_dialogue, add_special_tokens=True, padding=True,
        #                                 return_tensors="pt")
        decoder_labels = self.tokenizer(decoder_caption_dialogue, add_special_tokens=False, padding=True,
                                        return_tensors="pt")

        decoder_attention_mask = decoder_labels.attention_mask
        decoder_labels = decoder_labels.input_ids

        # <s><s><sty>style<utr>utterance
        decoder_input_ids = shift_tokens_right(decoder_labels, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id)
        decoder_attention_mask[:, 1:] = decoder_attention_mask[:, :-1].clone()
        decoder_attention_mask[:, 0] = 1

        return TestData(
            encoder_input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_labels=decoder_labels,
            decoder_str_labels=decoder_caption_dialogue,
            image_feature=image_features,
            test_turn=self.test_turn
        )


class GenerateDataForPreTraining():
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, batch):
        captions = []
        image_features = []
        dialogues = []
        for caption, image, dialogue in batch:
            captions.append(caption + "</s>")  #
            image_features.append(
                image.unsqueeze(0))  # [torch.FloatTensor: (1, 50, 512), torch.FloatTensor: (1, 50, 512), ... ]

            tmp = []
            for turn in dialogue:
                tmp.append(turn + "</s>")
            dialogues.append(
                tmp)  # [[<utr>style<sty>turn1, <utr>style<sty>turn2, <utr>style<sty>turn3], [style<sty>turn1, style<sty>turn2, style<sty>turn3] ... [turn1, turn2, turn3]]

        image_features = torch.vstack(image_features)

        change_cap_data_output = change_caption_data(captions) if 'ccd' in self.config.task_list else None
        image_swapping_output = image_swapping(image_features) if 'iso' in self.config.task_list else None
        change_dialogue_output = change_dialogue_data(dialogues)
        change_order_dialogue_output = change_order_dialogues(dialogues,
                                                              return_type="pt") if 'cod' in self.config.task_list else None
        ccd_caption_dialogues = []
        iso_caption_dialogues = []
        cdd_caption_dialogues = []
        cod_caption_dialogues = []

        for idx, caption_changed in enumerate(captions):
            if 'ccd' in self.config.task_list:
                ccd_caption_dialogues.append(
                    change_cap_data_output.unmodified_indices[idx] + change_dialogue_output.original_dialgoues[
                        idx] + "</s>")

            if 'iso' in self.config.task_list:
                iso_caption_dialogues.append(
                    caption_changed + change_dialogue_output.original_dialgoues[idx] + "</s>")

            if 'cdd' in self.config.task_list:
                cdd_caption_dialogues.append(
                    caption_changed + change_dialogue_output.modified_dialogues[idx] + "</s>")

            if 'cod' in self.config.task_list:
                cod_caption_dialogues.append(
                    caption_changed + change_order_dialogue_output.modified_dialogues[idx] + "</s>")

        if 'ccd' in self.config.task_list:
            ccd_caption_dialogues = self.tokenizer(ccd_caption_dialogues, add_special_tokens=False, padding=True,
                                                   return_tensors="pt").input_ids
        # |ccd_caption_dialogues| = (bsz, max_len)
        if 'iso' in self.config.task_list:
            iso_caption_dialogues = self.tokenizer(iso_caption_dialogues, add_special_tokens=False, padding=True,
                                                   return_tensors="pt").input_ids
        # |iso_caption_dialogues| = (bsz, max_len)

        if 'cdd' in self.config.task_list:
            cdd_caption_dialogues = self.tokenizer(cdd_caption_dialogues, add_special_tokens=False, padding=True,
                                                   return_tensors="pt").input_ids
        # |cdd_caption_dialogues| = (bsz, max_len)

        if 'cod' in self.config.task_list:
            cod_caption_dialogues = self.tokenizer(cod_caption_dialogues, add_special_tokens=False, padding=True,
                                                   return_tensors="pt").input_ids
        # |cod_caption_dialogues| = (bsz, max_len)

        return TrainingInputData(
            ccd_caption_dialogues=ccd_caption_dialogues,
            ccd_caption_dialogues_labels=change_cap_data_output.labels if change_cap_data_output is not None else None,
            # Task1
            iso_caption_dialogues=iso_caption_dialogues,  # torch.FloatTensor (bsz, 50, 512)
            image_features=image_features,
            image_features_swapped=image_swapping_output.modified_images,
            image_features_swapped_labels=image_swapping_output.labels if image_swapping_output is not None else None,
            # Task2
            cdd_caption_dialogues=cdd_caption_dialogues,
            cdd_caption_dialogues_labels=change_dialogue_output.labels if change_dialogue_output is not None else None,
            # Task3
            cod_caption_dialogues=cod_caption_dialogues,
            cod_caption_dialogues_labels=change_order_dialogue_output.labels if change_order_dialogue_output is not None else None,
            # Task4
        )