import torch
import torch.nn as nn
from .config import *
from .dataclass import *

from typing import Union, Optional, Tuple, List, Dict, Set
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from transformers import BartTokenizer, BartForConditionalGeneration, BartModel

mapping_tokens = {
        "<img>" : "image",
        # "<utr>" : "dialogue",
        # "<cap>" : "caption",
        "</img>" : "image",
        "<sty>" : "style"
}

mapping_tokens_for_config = {
        "<img>" : "img_bos_token_id",
        "</img>" : "img_eos_token_id",
        "<mrm>" : "mrm_token_id",
        "<sty>" : "sty_token_id",
        "<cid>" : "cid_token_id",
        "<gcd>" : "gcd_token_id",
        "<mlm>" : "mlm_token_id",
        "<feat>" : "feat_token_id",
        "<zero>" : "zero_token_id"
}

def add_new_token_to_tokenizer(
        new_tokens: Union[List[str]],
        tokenizer,
        model) -> Set[str]:
    # new tokens
    # img_bos_token, img_eos_token, utr_token, cap_token, mcm_token, cid_token, cdd_token, mlm_token, cdo_token
    new_tokens = new_tokens  # ["<img>", "</img>", "<utr>", "<cap>", "<mcm>", "<cid>", "<cdd>", "<mlm>", "<cdo>"]

    # check if the tokens are already in the vocabulary
    new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(new_tokens))

    # add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer))
    return new_tokens


def assign_embedding_space_for_a_new_token(
        new_tokens: Union[List[str]],
        model,
        tokenizer,
        embeddings,
        bos_token_id: int = 0,
        eos_token_id: int = 2
):
    start_of_tokens = ["<img>", "<utr>", "<cap>", "<mcm>" , "<cid>", "<cdd>", "<mlm>", "<cod>"]
    end_of_tokens = ["</img>"]

    for new_tok in new_tokens:
        if new_tok in start_of_tokens:
            with torch.no_grad():
                new_tok_idx = tokenizer.convert_tokens_to_ids(new_tok)
                embeddings.weight[new_tok_idx] = embeddings.weight[bos_token_id]

                # for tok in tokenizer(mapping_tokens[new_tok], special_tokens=False)['input_ids']:
                #     embeddings.weight[new_tok_idx] += embeddings.weight[tok]
        elif new_tok in end_of_tokens:
            with torch.no_grad():
                new_tok_idx = tokenizer.convert_tokens_to_ids(new_tok)
                embeddings.weight[new_tok_idx] = embeddings.weight[eos_token_id]

                # for tok in tokenizer(mapping_tokens[new_tok], special_tokens=False)['input_ids']:
                #     embeddings.weight[new_tok_idx] += embeddings.weight[tok]

    model.set_input_embeddings(embeddings)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class ImageEmbedding(nn.Module):
    def __init__(self, image_dim, final_dim):
        super(ImageEmbedding, self).__init__()
        self.linear = nn.Linear(image_dim, final_dim)
        self.layer_norm = nn.LayerNorm(final_dim)

    def forward(self, image_features):
        return self.layer_norm(self.linear(image_features))
#
# # ImageSwapping
# class MultiModalBartEncoder_ISO(nn.Module):
#     def __init__(self, config: MultiModalBartConfig):
#         super(MultiModalBartEncoder_ISO, self).__init__()
# 
#         self.config=config
#         self.iso_head = BartClassificationHead(config.d_model,
#                                                config.d_model,
#                                                config.iso_num_labels,
#                                                config.classif_dropout)
# 
#         self._init_weights(self.iso_head.dense)
#         self._init_weights(self.iso_head.out_proj)
# 
#     def _init_weights(self, module):
#         module.weight.data.normal_(mean=0.0, std=0.02)
#         if module.bias is not None:
#             module.bias.data.zero_()
# 
#     def forward(self,
#                 hidden_states: torch.FloatTensor
#     )->torch.FloatTensor:
#         # |hidden_states| = (bsz, image_length + cap_length + utrs_length, 768)
#         return self.iso_head(hidden_states)
# 
# # ChangeCaptionData
# class MultiModalBartEncoder_CCDO(nn.Module):
#     def __init__(self, config: MultiModalBartConfig):
#         super(MultiModalBartEncoder_CCDO, self).__init__()
# 
#         self.config=config
#         self.ccdo_head = BartClassificationHead(config.d_model,
#                                                config.d_model,
#                                                config.ccdo_num_labels,
#                                                config.classif_dropout)
# 
#         self._init_weights(self.ccdo_head.dense)
#         self._init_weights(self.ccdo_head.out_proj)
# 
#     def _init_weights(self, module):
#         module.weight.data.normal_(mean=0.0, std=0.02)
#         if module.bias is not None:
#             module.bias.data.zero_()
# 
#     def forward(self,
#                 hidden_states: torch.FloatTensor
#     )->torch.FloatTensor:
#         # |hidden_states| = (bsz, image_length + cap_length + utrs_length, 768)
#         return self.ccdo_head(hidden_states)
# 
# # ChangeCaptionDataOutput
# # class MultiModalBartEncoder_CCDO(nn.Module):
# #     def __init__(self, config: MultiModalBartConfig):
# #         super(MultiModalBartEncoder_CCDO, self).__init__()
# #
# #         self.config=config
# #         self.ccdo_head = BartClassificationHead(config.d_model,
# #                                                config.d_model,
# #                                                config.ccdo_num_labels,
# #                                                config.classif_dropout)
# #
# #         self._init_weights(self.ccdo_head.dense)
# #         self._init_weights(self.ccdo_head.out_proj)
# #
# #     def _init_weights(self, module):
# #         module.weight.data.normal_(mean=0.0, std=0.02)
# #         if module.bias is not None:
# #             module.bias.data.zero_()
# #
# #     def forward(self,
# #                 hidden_states: torch.FloatTensor
# #     )->torch.FloatTensor:
# #         # |hidden_states| = (bsz, image_length + cap_length + utrs_length, 768)
# #         return self.ccdo_head(hidden_states)
# 
# # ChangeDialogueDataOutput
# class MultiModalBartEncoder_CDDO(nn.Module):
#     def __init__(self, config: MultiModalBartConfig):
#         super(MultiModalBartEncoder_CDDO, self).__init__()
# 
#         self.config=config
#         self.cddo_head = BartClassificationHead(config.d_model,
#                                                config.d_model,
#                                                config.cddo_num_labels,
#                                                config.classif_dropout)
# 
#         self._init_weights(self.cddo_head.dense)
#         self._init_weights(self.cddo_head.out_proj)
# 
#     def _init_weights(self, module):
#         module.weight.data.normal_(mean=0.0, std=0.02)
#         if module.bias is not None:
#             module.bias.data.zero_()
# 
#     def forward(self,
#                 hidden_states: torch.FloatTensor
#     )->torch.FloatTensor:
#         # |hidden_states| = (bsz, image_length + cap_length + utrs_length, 768)
#         return self.cddo_head(hidden_states)
# 
# # ChangeOrderDialoguesOutput
# class MultiModalBartEncoder_CODO(nn.Module):
#     def __init__(self, config: MultiModalBartConfig):
#         super(MultiModalBartEncoder_CODO, self).__init__()
# 
#         self.config=config
#         self.codo_head = BartClassificationHead(config.d_model,
#                                                config.d_model,
#                                                config.codo_num_labels,
#                                                config.classif_dropout)
# 
#         self._init_weights(self.codo_head.dense)
#         self._init_weights(self.codo_head.out_proj)
# 
#     def _init_weights(self, module):
#         module.weight.data.normal_(mean=0.0, std=0.02)
#         if module.bias is not None:
#             module.bias.data.zero_()
# 
#     def forward(self,
#                 hidden_states: torch.FloatTensor
#     )->torch.FloatTensor:
#         # |hidden_states| = (bsz, image_length + cap_length + utrs_length, 768)
#         return self.codo_head(hidden_states)
# 
# class MultiModalBartDecoder(nn.Module):
#     def __init__(self, config, decoder):
#         super().__init__()
#         self.config = config
# 
#         self.dropout = decoder.dropout
#         self.layerdrop = decoder.layerdrop
# 
#         self.padding_idx = decoder.padding_idx
# 
#         self.eos_token_id = config.eos_token_id
# 
#         self.max_source_positions = decoder.max_source_positions
#         self.embed_scale = decoder.embed_scale
# 
#         self.embed_tokens = decoder.embed_tokens
#         self.embed_positions = decoder.embed_positions
# 
#         self.layers = decoder.layers
#         self.layernorm_embedding = decoder.layernorm_embedding
# 
#     def forward(
#             self,
#             caption_dialogues: torch.LongTensor = None,
#             encoder_hidden_states: torch.FloatTensor = None,
#             encoder_attention_mask: torch.FloatTensor = None,
#             output_attentions: Optional[bool] = False,
#             output_hidden_states: Optional[bool] = False,
#             return_dict: Optional[bool] = True,
#     ) -> BaseModelOutput:
#         """
# 
# 
#         :param caption_dialogues:
#         :param encoder_hidden_states:
#         :param encoder_attention_mask:
#         :param output_attentions:
#         :param output_hidden_states:
#         :param return_dict:
#         :return:
#         """
#         # |caption_dialogues| = (bsz, seq_len)
#         caption_dialogues_embed = self.embed_tokens(caption_dialogues)
#         # |caption_dialogues_embed| = (bsz, seq_len, d_model)
# 
#         bsz = caption_dialogues.shape[0]
#         tgt_len = caption_dialogues.shape[1]
# 
#         attention_mask = caption_dialogues.new_zeros(size=(bsz, tgt_len))
#         # |attention_mask| = (bsz, src_len)
#         attention_mask[caption_dialogues == self.config.pad_token_id] = torch.finfo(caption_dialogues_embed.dtype).min
#         attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, tgt_len).to(caption_dialogues_embed.dtype).to("cuda")
#         # |attention_mask| = (bsz, 1, src_len, src_len)
#         pass
# class MultiModalBartEncoder(nn.Module):
#     def __init__(self, config, encoder):
#         super().__init__()
#         self.config = config
# 
#         self.dropout = encoder.dropout
#         self.layerdrop = encoder.layerdrop
# 
#         self.padding_idx = encoder.padding_idx
#         self.img_bos_token_id = config.img_bos_token_id
#         self.img_eos_token_id = config.img_eos_token_id
#         self.cap_token_id = config.cap_token_id
#         self.utr_token_id = config.utr_token_id
# 
#         self.max_source_positions = encoder.max_source_positions
#         self.embed_scale = encoder.embed_scale
# 
#         self.embed_tokens = encoder.embed_tokens
#         self.embed_images = ImageEmbedding(self.config.image_feature_size, self.config.d_model)
#         self.modal_embed = nn.Embedding(2, self.config.d_model)
#         self.embed_positions = encoder.embed_positions
# 
#         self.layers = encoder.layers
#         self.layernorm_embedding = encoder.layernorm_embedding

#     def _embed_multi_modal(self,
#                            image_features: torch.FloatTensor,
#                            caption_dialogues: torch.LongTensor) -> EmbedMultiModalOutput:
#         """embed textual and visual inputs and combine them into one embedding"""
#         # |image_fueates| = (bsz, 50, 512)
#         bsz, _ = caption_dialogues.size()
#         caption_dialogues = caption_dialogues.to("cuda")
# 
#         image_features_embeds = self.embed_images(image_features)
#         # |image_feature_embeds| = (bsz, 9, 768)
#         # img_bos_token_embed = self.embed_tokens(torch.LongTensor(image_features.new_full(size=(bsz, 1), fill_value=self.config.img_bos_token_id)))
#         img_bos_token_embed = self.embed_tokens(torch.tensor(self.config.img_bos_token_id).to("cuda")).unsqueeze(0).unsqueeze(0)
#         img_bos_token_embed = img_bos_token_embed.repeat(bsz, 1, 1)
#         # |img_bos_token_embed| = (bsz, 1, 768)
#         # img_eos_token_embed = self.embed_tokens(torch.LongTensor(image_features.new_full(size=(bsz, 1), fill_value=self.config.img_eos_token_id)))
#         img_eos_token_embed = self.embed_tokens(torch.tensor(self.config.img_eos_token_id).to("cuda")).unsqueeze(0).unsqueeze(0)
#         img_eos_token_embed = img_eos_token_embed.repeat(bsz, 1, 1)
#         # |img_eos_token_embed| = (bsz, 1, 768)
#         image_features_embeds = torch.cat([img_bos_token_embed, image_features_embeds, img_eos_token_embed], dim=1)
#         image_length = image_features_embeds.shape[1]
#         # |image_feature_embeds| = (bsz, 11, 768)
# 
#         caption_dialogues_embed = self.embed_tokens(caption_dialogues)
#         # |input_cap_embed| = (bsz, seq_len, 768)
# 
#         embed = torch.cat([image_features_embeds, caption_dialogues_embed], dim=1)
#         # |embed| = (bsz, image_length + caption_dialogues_length, 768)
# 
#         img_idx = caption_dialogues.new_zeros((bsz, image_features_embeds.shape[1]))
#         caption_dialogues = torch.cat([img_idx, caption_dialogues], dim=1)  # |caption_dialogues| = (bsz, all_length)
# 
#         # attention_mask = embed.new_zeros(size=(bsz, embed.shape[1]))
#         # # |attention_mask| = (bsz, all_length)
#         # src_len = attention_mask.shape[-1]
#         # attention_mask[caption_dialogues == self.config.pad_token_id] = torch.finfo(embed.dtype).min
#         # attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, src_len, src_len).to(embed.dtype).to("cuda")
#         # attention_mask = attention_mask.masked_fill((attention == self.config.pad_token_id), torch.finfo(embed.dtype).min).to("cuda")
#         return EmbedMultiModalOutput(
#             image_length=image_length,
#             inputs_embeds=embed,  # |embed| = (bsz, image_length + cap_length + utrs_length, 768)
#             # attention_mask=attention_mask, # |attention_mask| = (bsz, 1, all_length, all_length)
#         )
# 
#     def forward(
#             self,
#             caption_dialogues: torch.LongTensor = None,
#             image_features: torch.FloatTensor = None,
#             attention_mask: torch.LongTensor = None,
#             output_attentions: Optional[bool] = False,
#             output_hidden_states: Optional[bool] = False,
#             return_dict: Optional[bool] = True
#     ) -> Union[Tuple[BaseModelOutput, EmbedMultiModalOutput]]:
#         embed_multi_modal_output = self._embed_multi_modal(image_features, caption_dialogues)
#         inputs_embeds = embed_multi_modal_output.inputs_embeds * self.embed_scale
# 
#         if attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
# 
#         embed_pos = self.embed_positions(inputs_embeds)
#         embed_pos = embed_pos.to(inputs_embeds.device)
# 
#         modal_embed_pos = embed_pos.new_zeros(size=(inputs_embeds.shape[0], inputs_embeds.shape[1]))
#         # |modal_embed_pos| = (bsz, all_length)
#         modal_embed_pos[:, embed_multi_modal_output.image_length:] = 1
#         modal_embed_pos = self.modal_embed(modal_embed_pos.long())
#         modal_embed_pos = modal_embed_pos.to(inputs_embeds.device)
#         # |modal_embed_pos| = (bsz, all_length, 767)
# 
#         hidden_states = inputs_embeds + embed_pos + modal_embed_pos
#         hidden_states = self.layernorm_embedding(hidden_states)
#         hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
#         # |hidden_states| = (bsz, image_length + cap_length + utrs_length, 768)
# 
#         # expand attention_mask
#         # attention_mask = embed_multi_modal_output.attention_mask
# 
#         encoder_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None
# 
#         for idx, encoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 encoder_states = encoder_states + (hidden_states,)
#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             dropout_probability = random.uniform(0, 1)
#             if self.training and (dropout_probability < self.layerdrop):  # skip the layer
#                 layer_outputs = (None, None)
#             else:
#                 # layer_outputs = (current_hidden_state, current_attention)
#                 layer_outputs = encoder_layer(
#                     hidden_states,
#                     attention_mask,
#                     output_attentions=output_attentions,
#                     layer_head_mask = (None)
#                 )
#                 hidden_states = layer_outputs[0]
# 
# 
#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)
# 
#         if output_hidden_states:
#             encoder_states = encoder_states + (hidden_states,)
# 
#         # if not return_dict:
#         #     return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
# 
#         return BaseModelOutput(
#             last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
#         )
# 
