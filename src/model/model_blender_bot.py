import torch
import gc
import torch.nn.functional as F
import numpy as np
import contractions

import torch.nn as nn

from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, F1Score
from einops import rearrange

from .modules import *
from .candidate_penalty_ce_loss import *
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.models.blenderbot import *
from transformers.models.blenderbot.modeling_blenderbot import *
from ..lightning.lightning import LightningBaseModel
from ..utils import compute_f1

import re

from torchmetrics.functional import bleu_score, rouge_score

_CONFIG_FOR_DOC = "BlenderbotConfig"

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class MultiModalBlenderbotEncoder(BlenderbotPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BlenderbotEncoderLayer`].

    Args:
        config: BlenderbotConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BlenderbotConfig, encoder, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.image_buffer = None
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.img_bos_token_id = config.img_bos_token_id
        self.img_eos_token_id = config.img_eos_token_id
        self.sep_token_id = config.sep_token_id

        self.max_source_positions = config.max_position_embeddings
        print("config.max_position_embeddings -> ", config.max_position_embeddings)
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_images = ImageEmbedding(self.config.image_feature_size, embed_dim)
        self.modal_embed = nn.Embedding(2, embed_dim)
        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        print("self.embed_positions -> (*)", self.embed_positions)
        self.layers = nn.ModuleList([BlenderbotEncoderLayer(config) for _ in range(config.encoder_layers)]) if encoder is None else encoder.layers
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()

    def set_image_buffer(self, images):
        self.image_buffer=images

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size() # (batch_size, sequence_length)
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_shape)

        image_embeds = self.embed_images(self.image_buffer)
        # |image_embeds| = (bsz, image_len, d_model)
        bsz, _, _ = image_embeds.size()
        # img_bos_token_embed = self.embed_tokens(torch.LongTensor(image_features.new_full(size=(bsz, 1), fill_value=self.config.img_bos_token_id)))
        img_bos_token_embed = self.embed_tokens(torch.tensor(self.config.img_bos_token_id).to(image_embeds.device)).unsqueeze(
            0).unsqueeze(0)
        img_bos_token_embed = img_bos_token_embed.repeat(bsz, 1, 1)
        img_eos_token_embed = self.embed_tokens(torch.tensor(self.config.img_eos_token_id).to(image_embeds.device)).unsqueeze(
            0).unsqueeze(0)
        img_eos_token_embed = img_eos_token_embed.repeat(bsz, 1, 1)
        image_embeds = torch.cat([img_bos_token_embed, image_embeds, img_eos_token_embed], dim=1)
        image_length = image_embeds.shape[1]
        # 여기 부분 수정해야함
        hidden_states = inputs_embeds + embed_pos
        hidden_states = torch.cat([image_embeds, hidden_states], dim = 1)
        modal_embed_pos = embed_pos.new_zeros(size=(hidden_states.shape[0], hidden_states.shape[1]))
        # |modal_embed_pos| = (bsz, all_length)
        modal_embed_pos[:, image_length:] = 1
        modal_embed_pos = self.modal_embed(modal_embed_pos.long())
        modal_embed_pos = modal_embed_pos.to(inputs_embeds.device)
        hidden_states = hidden_states + modal_embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # add final layer norm
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class MultiModalBlenderbotModel(BlenderbotModel):

    def build_model(self, config):
        """
            build_model 함수에서는 모델에 필요한 요소들을 정의해서 반환해준다.
        :param config:
        :return:
        """

        blender_model = AutoModel.from_pretrained(config.pretrained_blender_model)
        blender_tokenizer = AutoTokenizer.from_pretrained(config.pretrained_blender_model)

        # Add a new token and assign embedding space.
        new_tokens = [config.img_bos_token, config.img_eos_token, config.mlm_token, config.cid_token, config.mrm_token,
                      config.gcd_token, config.sty_token, config.feat_token, config.zero_token]
        # In the 'add_new_token_to_tokenizer' function, the newly defined 'new_tokens' above are added to the tokenizer, and they are also reflected in the model.
        new_tokens = add_new_token_to_tokenizer(new_tokens, blender_tokenizer, blender_model)
        # In 'assigin_embedding_space_for_a_new_tokne', it assigns embedding values for a new token.
        # assign_embedding_space_for_a_new_token(new_tokens, bart_model, bart_tokenizer, bart_model.get_input_embeddings(), bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id)

        for tok in new_tokens:
            tok_id = blender_tokenizer.convert_tokens_to_ids(tok)
            setattr(config, mapping_tokens_for_config[tok], tok_id)

        encoder = blender_model.encoder
        multimodal_blender_encoder = MultiModalBlenderbotEncoder(config=config, encoder=encoder, embed_tokens = blender_model.shared)

        if config.max_position_embeddings > 128:
            with torch.no_grad():
                multimodal_blender_encoder.embed_positions.weight[:128] = encoder.embed_positions.weight[:]
                cnt = 1
                for i in range(128, 256):
                    multimodal_blender_encoder.embed_positions.weight[i] = encoder.embed_positions.weight[i-cnt] + encoder.embed_positions.weight[0]
                    cnt+=1

            blender_model.decoder.embed_positions = multimodal_blender_encoder.embed_positions
        else:
            multimodal_blender_encoder.embed_positions = encoder.embed_positions

        multimodal_blender_encoder.layer_norm = encoder.layer_norm

        return multimodal_blender_encoder, blender_model.decoder, blender_tokenizer
    def __init__(self, config: BlenderbotConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, 8018
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder, self.decoder, self.tokenizer = self.build_model(config)
        self.shared = self.encoder.embed_tokens

        print("self.encoder.embed_positions ->",self.encoder.embed_positions)
        print("self.decoder.embed_positions ->", self.decoder.embed_positions)
        # self.encoder, self.decoder, self.tokenizer = self.build_model(config)
        # self.encoder = MultiModalBlenderbotEncoder(config, self.shared)
        # self.decoder = BlenderbotDecoder(config, self.shared)
        # self.post_init()

    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import BlenderbotTokenizer, BlenderbotModel

            >>> model = BlenderbotModel.from_pretrained("facebook/blenderbot-400M-distill")
            >>> tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class MultiModalBlendPretraining(BlenderbotForConditionalGeneration, LightningBaseModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder.version",
        r"decoder.version",
        r"lm_head.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_tokens.weight",
    ]

    def __init__(self, config: BlenderbotConfig):
        super().__init__(config)
        self.model = MultiModalBlenderbotModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.mrm_head = nn.Linear(config.d_model, config.image_feature_size)

    def training_step(self, batch, batch_idx):
        gc.collect()
        torch.cuda.empty_cache()
        training_input_data = batch

        losses = 0

        metrics = {}
        for task in self.config.task_list:
            if task == 'mlm':
                self.model.encoder.set_image_buffer(training_input_data.image_feature)
                model_output = self.model(
                    input_ids=training_input_data.mlm_caption_dialogues,
                    attention_mask=training_input_data.mlm_attention_mask,
                    decoder_input_ids=training_input_data.mlm_decoder_inputs,
                    decoder_attention_mask=training_input_data.mlm_decoder_attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )

                # print("\n")
                # print(self.model.tokenizer.batch_decode(training_input_data.mlm_caption_dialogues))
                # print(self.model.tokenizer.batch_decode(training_input_data.mlm_decoder_inputs))
                # print(self.model.tokenizer.batch_decode(training_input_data.mlm_decoder_labels))

                lm_logits = self.lm_head(model_output.last_hidden_state)
                lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
                # |lm_logits| = (bsz, tgt_len, vocab_size)

                if training_input_data.mlm_decoder_labels is not None:
                    loss_fct = CrossEntropyLoss(ignore_index=self.model.tokenizer.pad_token_id)
                    masked_lm_loss = loss_fct(rearrange(lm_logits, "b t v -> (b t) v"),
                                              training_input_data.mlm_decoder_labels.view(-1))
                metrics['mlm_perplexity'] = torch.exp(masked_lm_loss.detach().cpu())
                masked_lm_loss = candidate_penalty_cross_entropy_criterion(lm_logits,
                                                                           training_input_data.mlm_decoder_labels,
                                                                           self.model.tokenizer)

                metrics['mlm_loss'] = masked_lm_loss.detach().cpu()
                losses += (masked_lm_loss * (1 / len(self.config.task_list)))
                # print("MLM Success")
            elif task == 'mrm':
                self.model.encoder.set_image_buffer(training_input_data.mrm_image_feature)

                model_output = self.model(
                    input_ids=training_input_data.mrm_caption_dialogues,
                    attention_mask=training_input_data.mrm_attention_mask,
                    decoder_input_ids=training_input_data.mrm_decoder_inputs,
                    decoder_attention_mask=training_input_data.mrm_decoder_attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )
                lm_logits = self.mrm_head(model_output.last_hidden_state)
                # lm_logits = F.log_softmax(lm_logits, dim=-1)
                # print("lm_logits ->",lm_logits.shape)
                # print(f'{lm_logits[training_input_data.mrm_decoder_inputs == self.model.tokenizer.convert_tokens_to_ids("<zero>")].shape}')
                # print(f'{training_input_data.image_feature[training_input_data.mrm_decoder_inputs[:, 2:] == self.model.tokenizer.convert_tokens_to_ids("<zero>")].shape}')
                # criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
                # loss = criterion(
                #     F.log_softmax(lm_logits[training_input_data.mrm_decoder_inputs == self.model.tokenizer.convert_tokens_to_ids("<zero>")], dim = -1),
                #     F.log_softmax(training_input_data.image_feature[training_input_data.mrm_decoder_inputs[:, 2:] == self.model.tokenizer.convert_tokens_to_ids("<zero>")], dim = -1)
                # )
                criterion = torch.nn.KLDivLoss(reduction='batchmean')
                loss = criterion(
                    F.log_softmax(lm_logits[training_input_data.mrm_decoder_inputs == self.model.tokenizer.convert_tokens_to_ids("<zero>")].double(), dim = -1),
                    F.softmax(training_input_data.image_feature[training_input_data.mrm_decoder_inputs[:, 2:] == self.model.tokenizer.convert_tokens_to_ids("<zero>")].double(), dim = -1)
                )
                metrics['mrm_loss'] = loss.detach().cpu()
                losses += (loss * (1 / len(self.config.task_list)))
                # print("MRM Success")
            elif task == 'cid':
                self.model.encoder.set_image_buffer(training_input_data.modified_image_feature)
                model_output = self.model(
                    input_ids=training_input_data.cid_caption_dialogues,
                    attention_mask=training_input_data.cid_attention_mask,
                    decoder_input_ids=training_input_data.cid_decoder_inputs,
                    decoder_attention_mask=training_input_data.cid_decoder_attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )
                lm_logits = self.lm_head(model_output.last_hidden_state)
                lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
                # |lm_logits| = (bsz, tgt_len, vocab_size)

                if training_input_data.cid_decoder_labels is not None:
                    loss_fct = CrossEntropyLoss(ignore_index=self.model.tokenizer.pad_token_id)
                    masked_lm_loss = loss_fct(rearrange(lm_logits, "b t v -> (b t) v"),
                                              training_input_data.cid_decoder_labels.view(-1))

                metrics['cid_loss'] = masked_lm_loss.detach().cpu()
                metrics['cid_perplexity'] = torch.exp(masked_lm_loss.detach().cpu())
                losses += (masked_lm_loss * (1 / len(self.config.task_list)))
                # print("CID Success")
            elif task == 'gcd':
                self.model.encoder.set_image_buffer(training_input_data.image_feature)
                model_output = self.model(
                    input_ids=training_input_data.gcd_caption_dialogues,
                    attention_mask=training_input_data.gcd_attention_mask,
                    decoder_input_ids=training_input_data.gcd_decoder_inputs,
                    decoder_attention_mask=training_input_data.gcd_decoder_attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )
                lm_logits = self.lm_head(model_output.last_hidden_state)
                lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
                # |lm_logits| = (bsz, tgt_len, vocab_size)

                if training_input_data.gcd_decoder_labels is not None:
                    loss_fct = CrossEntropyLoss(ignore_index=self.model.tokenizer.pad_token_id)
                    masked_lm_loss = loss_fct(rearrange(lm_logits, "b t v -> (b t) v"),
                                              training_input_data.gcd_decoder_labels.view(-1))
                metrics['gcd_perplexity'] = torch.exp(masked_lm_loss.detach().cpu())
                masked_lm_loss = candidate_penalty_cross_entropy_criterion(lm_logits,
                                                                           training_input_data.gcd_decoder_labels,
                                                                           self.model.tokenizer)

                metrics['gcd_loss'] = masked_lm_loss.detach().cpu()
                losses += (masked_lm_loss * (1 / len(self.config.task_list)))
                # print("GCD Success")

        self.log_dict(metrics, prog_bar=True)

        return losses

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=6.25e-5, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=6.25e-5,
                                                        steps_per_epoch=self.config.steps_per_epoch,
                                                        epochs=self.config.max_epochs,
                                                        anneal_strategy='linear')
        return ([optimizer], [scheduler])

class MultiModalBlendFinetuning(BlenderbotForConditionalGeneration, LightningBaseModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder.version",
        r"decoder.version",
        r"lm_head.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_tokens.weight",
    ]

    def __init__(self, config: BlenderbotConfig):
        super().__init__(config)
        self.config = config
        self.model = MultiModalBlenderbotModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.mrm_head = nn.Linear(config.d_model, config.image_feature_size)
        self.turn_one_rougeL = []
        self.turn_one_bleu_4 = []
        self.turn_one_f1 = []
        self.turn_one_ppl = []

        self.turn_two_rougeL = []
        self.turn_two_bleu_4 = []
        self.turn_two_f1 = []
        self.turn_two_ppl = []

        self.turn_three_rougeL = []
        self.turn_three_bleu_4 = []
        self.turn_three_f1 = []
        self.turn_three_ppl = []

        self.labels = []
        self.predict = []
        self.image_hashes = []

        self.epoch = 0
        self.metrics = {}
        self.test_turn = None
    def training_step(self, batch, batch_idx):
        gc.collect()
        torch.cuda.empty_cache()

        training_input_data = batch

        metrics = {}

        self.model.encoder.set_image_buffer(training_input_data.image_feature)
        model_output = self.model(
            input_ids=training_input_data.encoder_input_ids,
            attention_mask=training_input_data.attention_mask,
            decoder_input_ids=training_input_data.decoder_input_ids,
            decoder_attention_mask=training_input_data.decoder_attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )


        lm_logits = self.lm_head(model_output.last_hidden_state)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        # |lm_logits| = (bsz, tgt_len, vocab_size)

        if training_input_data.decoder_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.model.tokenizer.pad_token_id)
            masked_lm_loss = loss_fct(rearrange(lm_logits, "b t v -> (b t) v"),
                                      training_input_data.decoder_labels.view(-1))
        metrics['train_ppl'] = torch.exp(masked_lm_loss)
        masked_lm_loss = candidate_penalty_cross_entropy_criterion(lm_logits, training_input_data.decoder_labels, self.model.tokenizer)


        metrics['train_loss'] = masked_lm_loss

        self.log_dict(metrics, prog_bar=True)

        return {'loss':masked_lm_loss,
                'train_perplexity':torch.exp(masked_lm_loss)}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        gc.collect()
        torch.cuda.empty_cache()
        training_input_data = batch

        self.test_turn = training_input_data.test_turn

        self.model.encoder.set_image_buffer(training_input_data.image_feature)
        self.image_hashes += training_input_data.image_hashes

        model_output = self.model(
            input_ids=training_input_data.encoder_input_ids,
            attention_mask=training_input_data.attention_mask,
            decoder_input_ids=training_input_data.decoder_input_ids,
            decoder_attention_mask=training_input_data.decoder_attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        lm_logits = self.lm_head(model_output.last_hidden_state)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        # |lm_logits| = (bsz, tgt_len, vocab_size)

        if training_input_data.decoder_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.model.tokenizer.pad_token_id)
            masked_lm_loss = loss_fct(rearrange(lm_logits, "b t v -> (b t) v"),
                                      training_input_data.decoder_labels.view(-1))
        ppl = torch.exp(masked_lm_loss).detach().cpu()

        if self.test_turn == "turn_one":
            self.turn_one_ppl.append(ppl)
        elif self.test_turn == "turn_two":
            self.turn_two_ppl.append(ppl)
        elif self.test_turn == "turn_three":
            self.turn_three_ppl.append(ppl)
        else:
            raise ValueError("Not Turn")


        model_output = self.generate(input_ids = training_input_data.encoder_input_ids,
                                     attention_mask = training_input_data.attention_mask,
                                     min_length = 20,
                                     num_beams=10)

        model_output = self.model.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        predict = [contractions.fix(text.strip().lower()) for text in model_output]
        # labels = self.model.tokenizer.batch_decode(training_input_data.decoder_labels, skip_special_tokens=True)
        labels = training_input_data.decoder_str_labels
        labels = self.model.tokenizer.batch_decode(self.model.tokenizer(labels, max_length=512, return_tensors="pt", padding=True, add_special_tokens=False).input_ids, skip_special_tokens=True)
        labels = [contractions.fix(text.strip().lower()) for text in labels]

        # print(f"\n\n @@@@@@@@@@@@@@ {self.test_turn} @@@@@@@@@@@@@@\n")
        # print("encoder_input_ids -> ", self.model.tokenizer.batch_decode(training_input_data.encoder_input_ids))
        # print("decoder_input_ids -> ", self.model.tokenizer.batch_decode(training_input_data.decoder_input_ids))
        # print("decoder_labels -> ", self.model.tokenizer.batch_decode(training_input_data.decoder_labels))
        # print("labels -> ", labels)
        # print("predict -> ", predict)

        self.labels += labels
        self.predict += predict

        rogue_scores = None
        bleu_scores = None
        try:
            rogue_scores = rouge_score(predict, labels)['rougeL_fmeasure']
        except Exception as e:
            rogue_scores = None
        try:
            bleu_scores = bleu_score(predict, labels)
        except Exception as e:
            bleu_scores = None

        f1_list = []
        for p, l in zip(predict, labels):
            f1 = compute_f1(l, p, self.model.tokenizer)
            f1_list.append(f1)

        f1_scores = 0
        for s in f1_list:
            f1_scores += (s / len(f1_list))


        if self.test_turn == "turn_one":
            if rogue_scores is not None:
                self.turn_one_rougeL.append(rogue_scores.cpu())
            if bleu_scores is not None:
                self.turn_one_bleu_4.append(bleu_scores.cpu())
            self.turn_one_f1.append(f1_scores)
        elif self.test_turn =="turn_two":
            if rogue_scores is not None:
                self.turn_two_rougeL.append(rogue_scores.cpu())
            if bleu_scores is not None:
                self.turn_two_bleu_4.append(bleu_scores.cpu())
            self.turn_two_f1.append(f1_scores)
        elif self.test_turn =="turn_three":
            if rogue_scores is not None:
                self.turn_three_rougeL.append(rogue_scores.cpu())
            if bleu_scores is not None:
                self.turn_three_bleu_4.append(bleu_scores.cpu())
            self.turn_three_f1.append(f1_scores)

    def on_validation_epoch_end(self):
        # self.metrics[f"{self.test_turn}_ppl"] = np.mean(torch.Tensor(self.ppl).numpy())
        # self.metrics[f"{self.test_turn}_rougeL"] = np.mean(torch.Tensor(self.rougeL).numpy())*100
        # self.metrics[f"{self.test_turn}_bleu_4"] = np.mean(torch.Tensor(self.bleu_4).numpy())*100
        # self.metrics[f"{self.test_turn}_f1"] = np.mean(torch.Tensor(self.f1).numpy())*100

        self.t1_ppl = np.mean(torch.Tensor(self.turn_one_ppl).numpy())
        self.t1_f1 = np.mean(torch.Tensor(self.turn_one_f1).numpy()) * 100
        self.t1_R = np.mean(torch.Tensor(self.turn_one_rougeL).numpy()) * 100
        self.t1_B = np.mean(torch.Tensor(self.turn_one_bleu_4).numpy()) * 100

        self.t2_ppl = np.mean(torch.Tensor(self.turn_two_ppl).numpy())
        self.t2_f1 = np.mean(torch.Tensor(self.turn_two_f1).numpy()) * 100
        self.t2_R = np.mean(torch.Tensor(self.turn_two_rougeL).numpy()) * 100
        self.t2_B = np.mean(torch.Tensor(self.turn_two_bleu_4).numpy()) * 100

        self.t3_ppl = np.mean(torch.Tensor(self.turn_three_ppl).numpy())
        self.t3_f1 = np.mean(torch.Tensor(self.turn_three_f1).numpy()) * 100
        self.t3_R = np.mean(torch.Tensor(self.turn_three_rougeL).numpy()) * 100
        self.t3_B = np.mean(torch.Tensor(self.turn_three_bleu_4).numpy()) * 100

        self.metrics['t1_ppl'] = self.t1_ppl
        self.metrics['t1_f1'] = self.t1_f1
        self.metrics['t1_B'] = self.t1_B
        self.metrics['t1_R'] = self.t1_R

        self.metrics['t2_ppl'] = self.t2_ppl
        self.metrics['t2_f1'] = self.t2_f1
        self.metrics['t2_B'] = self.t2_B
        self.metrics['t2_R'] = self.t2_R

        self.metrics['t3_ppl'] = self.t3_ppl
        self.metrics['t3_f1'] = self.t3_f1
        self.metrics['t3_B'] = self.t3_B
        self.metrics['t3_R'] = self.t3_R

        self.log_dict(self.metrics, prog_bar=True)

        with open(f"../checkpoints/label_and_predict/{self.epoch}_{self.test_turn}.txt", "w") as f:
            for i in range(len(self.labels)):
                f.write(f"label -> {self.labels[i]}\n")
                f.write(f"predict -> {self.predict[i]}\n")
                f.write(f"image_hash -> {self.image_hashes[i]}\n\n")

        self.epoch += 1

        self.turn_one_ppl.clear()
        self.turn_one_bleu_4.clear()
        self.turn_one_f1.clear()
        self.turn_one_rougeL.clear()

        self.turn_two_ppl.clear()
        self.turn_two_bleu_4.clear()
        self.turn_two_f1.clear()
        self.turn_two_rougeL.clear()

        self.turn_three_ppl.clear()
        self.turn_three_bleu_4.clear()
        self.turn_three_f1.clear()
        self.turn_three_rougeL.clear()

        self.labels.clear()
        self.image_hashes.clear()
        self.predict.clear()

    def test_step(self, batch, batch_idx, dataloader_idx):
        torch.cuda.empty_cache()
        training_input_data = batch

        self.test_turn = training_input_data.test_turn

        self.model.encoder.set_image_buffer(training_input_data.image_feature)

        model_output = self.model(
            input_ids=training_input_data.encoder_input_ids,
            attention_mask=training_input_data.attention_mask,
            decoder_input_ids=training_input_data.decoder_input_ids,
            decoder_attention_mask=training_input_data.decoder_attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        lm_logits = self.lm_head(model_output.last_hidden_state)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        # |lm_logits| = (bsz, tgt_len, vocab_size)

        if training_input_data.decoder_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.model.tokenizer.pad_token_id)
            masked_lm_loss = loss_fct(rearrange(lm_logits, "b t v -> (b t) v"),
                                      training_input_data.decoder_labels.view(-1))
        ppl = torch.exp(masked_lm_loss).detach().cpu()

        if self.test_turn == "turn_one":
            self.turn_one_ppl.append(ppl)
        elif self.test_turn == "turn_two":
            self.turn_two_ppl.append(ppl)
        elif self.test_turn == "turn_three":
            self.turn_three_ppl.append(ppl)
        else:
            raise ValueError("Not Turn")

        model_output = self.generate(input_ids=training_input_data.encoder_input_ids,
                                     attention_mask=training_input_data.attention_mask,
                                     min_length=20,
                                     num_beams=10)

        model_output = self.model.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        predict = [contractions.fix(text.strip().lower()) for text in model_output]
        # labels = self.model.tokenizer.batch_decode(training_input_data.decoder_labels, skip_special_tokens=True)
        labels = training_input_data.decoder_str_labels
        labels = self.model.tokenizer.batch_decode(
            self.model.tokenizer(labels, max_length=512, return_tensors="pt", padding=True,
                                 add_special_tokens=False).input_ids, skip_special_tokens=True)
        labels = [contractions.fix(text.strip().lower()) for text in labels]

        # print(f"\n\n @@@@@@@@@@@@@@ {self.test_turn} @@@@@@@@@@@@@@\n")
        # print("encoder_input_ids -> ", self.model.tokenizer.batch_decode(training_input_data.encoder_input_ids))
        # print("decoder_input_ids -> ", self.model.tokenizer.batch_decode(training_input_data.decoder_input_ids))
        # print("decoder_labels -> ", self.model.tokenizer.batch_decode(training_input_data.decoder_labels))
        # print("labels -> ", labels)
        # print("predict -> ", predict)

        self.labels += labels
        self.predict += predict

        rogue_scores = None
        bleu_scores = None
        try:
            rogue_scores = rouge_score(predict, labels)['rougeL_fmeasure']
        except Exception as e:
            rogue_scores = None
        try:
            bleu_scores = bleu_score(predict, labels)
        except Exception as e:
            bleu_scores = None

        f1_list = []
        for p, l in zip(predict, labels):
            f1 = compute_f1(l, p, self.model.tokenizer)
            f1_list.append(f1)

        f1_scores = 0
        for s in f1_list:
            f1_scores += (s / len(f1_list))

        if self.test_turn == "turn_one":
            if rogue_scores is not None:
                self.turn_one_rougeL.append(rogue_scores.cpu())
            if bleu_scores is not None:
                self.turn_one_bleu_4.append(bleu_scores.cpu())
            self.turn_one_f1.append(f1_scores)
        elif self.test_turn == "turn_two":
            if rogue_scores is not None:
                self.turn_two_rougeL.append(rogue_scores.cpu())
            if bleu_scores is not None:
                self.turn_two_bleu_4.append(bleu_scores.cpu())
            self.turn_two_f1.append(f1_scores)
        elif self.test_turn == "turn_three":
            if rogue_scores is not None:
                self.turn_three_rougeL.append(rogue_scores.cpu())
            if bleu_scores is not None:
                self.turn_three_bleu_4.append(bleu_scores.cpu())
            self.turn_three_f1.append(f1_scores)
    def on_test_epoch_end(self):
        self.t1_ppl = np.mean(torch.Tensor(self.turn_one_ppl).numpy())
        self.t1_f1 = np.mean(torch.Tensor(self.turn_one_f1).numpy()) * 100
        self.t1_R = np.mean(torch.Tensor(self.turn_one_rougeL).numpy()) * 100
        self.t1_B = np.mean(torch.Tensor(self.turn_one_bleu_4).numpy()) * 100

        self.t2_ppl = np.mean(torch.Tensor(self.turn_two_ppl).numpy())
        self.t2_f1 = np.mean(torch.Tensor(self.turn_two_f1).numpy()) * 100
        self.t2_R = np.mean(torch.Tensor(self.turn_two_rougeL).numpy()) * 100
        self.t2_B = np.mean(torch.Tensor(self.turn_two_bleu_4).numpy()) * 100

        self.t3_ppl = np.mean(torch.Tensor(self.turn_three_ppl).numpy())
        self.t3_f1 = np.mean(torch.Tensor(self.turn_three_f1).numpy()) * 100
        self.t3_R = np.mean(torch.Tensor(self.turn_three_rougeL).numpy()) * 100
        self.t3_B = np.mean(torch.Tensor(self.turn_three_bleu_4).numpy()) * 100

        self.metrics['t1_ppl'] = self.t1_ppl
        self.metrics['t1_f1'] = self.t1_f1
        self.metrics['t1_B'] = self.t1_B
        self.metrics['t1_R'] = self.t1_R

        self.metrics['t2_ppl'] = self.t2_ppl
        self.metrics['t2_f1'] = self.t2_f1
        self.metrics['t2_B'] = self.t2_B
        self.metrics['t2_R'] = self.t2_R

        self.metrics['t3_ppl'] = self.t3_ppl
        self.metrics['t3_f1'] = self.t3_f1
        self.metrics['t3_B'] = self.t3_B
        self.metrics['t3_R'] = self.t3_R

        self.log_dict(self.metrics, prog_bar=True)

        with open(f"../checkpoints/label_and_predict/{self.epoch}_{self.test_turn}.txt", "w") as f:
            for i in range(len(self.labels)):
                f.write(f"label -> {self.labels[i]}\n")
                f.write(f"predict -> {self.predict[i]}\n\n")

        self.epoch += 1

        self.turn_one_ppl.clear()
        self.turn_one_bleu_4.clear()
        self.turn_one_f1.clear()
        self.turn_one_rougeL.clear()

        self.turn_two_ppl.clear()
        self.turn_two_bleu_4.clear()
        self.turn_two_f1.clear()
        self.turn_two_rougeL.clear()

        self.turn_three_ppl.clear()
        self.turn_three_bleu_4.clear()
        self.turn_three_f1.clear()
        self.turn_three_rougeL.clear()

        self.labels.clear()
        self.predict.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        gc.collect()
        torch.cuda.empty_cache()
        training_input_data = batch

        metrics = {}

        self.model.encoder.set_image_buffer(training_input_data.image_feature)
        model_output = self.generate(input_ids=training_input_data.encoder_input_ids,
                                     attention_mask=training_input_data.attention_mask,
                                     min_length=20,
                                     num_beams=self.config.num_beams)

        model_output = self.model.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        predict = [contractions.fix(text.stirp().lower()) for text  in model_output]

        return predict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=6.25e-5, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=6.25e-5,
                                                        steps_per_epoch=self.config.steps_per_epoch,
                                                        epochs=self.config.max_epochs,
                                                        anneal_strategy='linear')
        return ([optimizer], [scheduler])

class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return 1.

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)
