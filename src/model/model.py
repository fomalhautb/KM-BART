# Based on transformers.modeling_bart

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_bart import (
    PretrainedBartModel,
    BartDecoder,
    BartClassificationHead,
    _make_linear_from_emb,
    _prepare_bart_decoder_inputs,
    _filter_out_falsey_values
)

from src.model.config import MultiModalBartConfig
from src.model.mixins import GenerationMixin, FromPretrainedMixin
from src.model.modules import MultiModalBartEncoder


# This is based on transformers.BartModel
# The modifications are:
# - BartConfig -> MultiModalBartConfig
# - BartEncoder -> MultiModalBartEncoder
# - added image_features in forward
class MultiModalBartModel(FromPretrainedMixin, PretrainedBartModel):
    def __init__(self, config: MultiModalBartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MultiModalBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def forward(
            self,
            input_ids,
            image_features,
            attention_mask=None,
            decoder_input_ids=None,
            encoder_outputs: Optional[Tuple] = None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                image_features=image_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        assert isinstance(encoder_outputs, tuple)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        return decoder_outputs + encoder_outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


# This is based on transformers.BartForConditionalGeneration
# The modifications are:
# - BartConfig -> MultiModalBartConfig
# - BartModel -> MultiModalBartModel
# - added image_features in forward
# - changed loss computation in forward
# - added image_features in prepare_inputs_for_generation
# - rewrite generate function
class MultiModalBartForPreTraining(FromPretrainedMixin, GenerationMixin, PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: MultiModalBartConfig):
        super().__init__(config)
        self.cls_token_id = config.cls_token_id
        self.model = MultiModalBartModel(config)

        self.mrm_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classif_dropout,
        )
        self._init_weights(self.mrm_head.dense)
        self._init_weights(self.mrm_head.out_proj)

        self.attribute_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_attributes,
            config.classif_dropout,
        )
        self._init_weights(self.attribute_head.dense)
        self._init_weights(self.attribute_head.out_proj)

        self.relation_head = BartClassificationHead(
            config.d_model * 2,
            config.d_model,
            config.num_relations,
            config.classif_dropout,
        )
        self._init_weights(self.relation_head.dense)
        self._init_weights(self.relation_head.out_proj)

        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def forward(
            self,
            input_ids,
            image_features,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            labels=None,
            mrm_labels=None,
            mrm_mask=None,
            attribute_labels=None,
            attribute_mask=None,
            relation_labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            **unused,
    ):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param encoder_outputs:
        :param decoder_input_ids:
        :param decoder_attention_mask:
        :param decoder_cached_states:
        :param labels: Labels for computing the language modeling loss.
            Indices should either be in [0, ..., config.vocab_size] or -100.
            Tokens with indices set to -100 are ignored (masked), the loss is only computed for the tokens
            with labels in [0, ..., config.vocab_size].
        :param mrm_labels: labels for computing masked region modeling loss. similar to labels
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param unused:
        :return: :obj:tuple(torch.FloatTensor) comprising various elements depending on the configuration and inputs:
            loss (optional, returned when labels is provided) dict of FloatTensor of shape (1,):
                {
                    'loss': total loss,
                    'lm_loss': Masked language modeling loss (if labels is given),
                    'mrm_loss': masked region modeling loss (if mrm_labels is given).
                }
            prediction_scores (:obj: torch.FloatTensor of shape :obj:(batch_size, sequence_length, config.vocab_size))
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            hidden_states (:obj:tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed
                or when config.output_hidden_states=True):
                Tuple of :obj:FloatTensor (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:(batch_size, sequence_length, hidden_size).

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed
                or when config.output_attentions=True):
                Tuple of :obj:FloatTensor (one for each layer) of shape
                :obj:(batch_size, num_heads, sequence_length, sequence_length).

                Attentions weights after the attention softmax, used to compute the weighted average
                in the self-attention heads.
        """

        if (labels is not None) or (mrm_labels is not None) or \
                (attribute_labels is not None) or (relation_labels is not None):
            use_cache = False

        if (mrm_labels is not None) and (mrm_mask is None):
            raise ValueError('"mrm_mask" cannot be None while "mrm_labels" is set')

        outputs = self.model(
            input_ids,
            image_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        losses = {}

        mrm_loss = 0
        # compute mrm loss if mrm_labels is given
        if mrm_labels is not None:
            region_representation = outputs[0][mrm_mask.bool()]
            # prevent mrm loss to be nan
            if len(region_representation) > 0:
                prediction_soft_label = self.mrm_head(region_representation)
                prediction_soft_label = F.log_softmax(prediction_soft_label, dim=1)
                mrm_labels = torch.cat(mrm_labels, dim=0)
                mrm_loss = F.kl_div(prediction_soft_label, mrm_labels, reduction='batchmean')
                mrm_loss *= self.config.mrm_loss_factor
                losses['mrm_loss'] = mrm_loss

        attribute_loss = 0
        if attribute_labels is not None:
            region_representation = outputs[0][attribute_mask.bool()]
            if len(region_representation) > 0:
                prediction_soft_label = self.attribute_head(region_representation)
                loss_fct = nn.CrossEntropyLoss()
                attribute_labels = torch.cat(attribute_labels, dim=0)
                attribute_loss = loss_fct(prediction_soft_label, attribute_labels.reshape(-1))
                attribute_loss *= self.config.attribute_loss_factor
                losses['attribute_loss'] = attribute_loss

        relation_loss = 0
        if relation_labels is not None:
            region_representation = []
            relation_label_ids = []
            for i, rels in enumerate(relation_labels):
                for rel in rels:
                    region_representation.append(torch.cat([
                        outputs[0][i][rel['object_index']],
                        outputs[0][i][rel['subject_index']]
                    ], axis=0))
                    relation_label_ids.append(rel['label'])

            if len(region_representation) > 0:
                region_representation = torch.stack(region_representation)
                prediction_soft_label = self.relation_head(region_representation)
                loss_fct = nn.CrossEntropyLoss()
                relation_labels = torch.LongTensor(relation_label_ids).to(prediction_soft_label.device)
                relation_loss = loss_fct(prediction_soft_label, relation_labels.reshape(-1))
                relation_loss *= self.config.relation_loss_factor
                losses['relation_loss'] = relation_loss

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here

        lm_loss = 0
        # compute lm loss if labels is given
        if labels is not None:
            labels = labels.clone()
            labels[labels == self.cls_token_id] = -100
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            lm_loss *= self.config.lm_loss_factor
            losses['lm_loss'] = lm_loss

        if (labels is not None) or (mrm_labels is not None) or (attribute_labels is not None) or (
                relation_labels is not None):
            losses['loss'] = lm_loss + mrm_loss + attribute_loss + relation_loss
            outputs = (losses,) + outputs

        return outputs


# This is based on transformers.BartForConditionalGeneration
# The modifications are:
# - BartConfig -> MultiModalBartConfig
# - BartModel -> MultiModalBartModel
# - added image_features in forward
class MultiModalBartForConditionalGeneration(FromPretrainedMixin, GenerationMixin, PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: MultiModalBartConfig):
        super().__init__(config)
        self.model = MultiModalBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def forward(
            self,
            input_ids,
            image_features,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            **unused,
    ):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param encoder_outputs:
        :param decoder_input_ids:
        :param decoder_attention_mask:
        :param decoder_cached_states:
        :param labels: Labels for computing the language modeling loss.
            Indices should either be in [0, ..., config.vocab_size] or -100.
            Tokens with indices set to -100 are ignored (masked), the loss is only computed for the tokens
            with labels in [0, ..., config.vocab_size].
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param unused:
        :return: :obj:tuple(torch.FloatTensor) comprising various elements depending on the configuration and inputs:
            loss (optional, returned when labels is provided) dict of FloatTensor of shape (1,):
                {
                    'loss': total loss,
                    'lm_loss': Masked language modeling loss (if labels is given),
                    'mrm_loss': masked region modeling loss (if mrm_labels is given).
                }
            prediction_scores (:obj: torch.FloatTensor of shape :obj:(batch_size, sequence_length, config.vocab_size))
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            hidden_states (:obj:tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed
                or when config.output_hidden_states=True):
                Tuple of :obj:FloatTensor (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:(batch_size, sequence_length, hidden_size).

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed
                or when config.output_attentions=True):
                Tuple of :obj:FloatTensor (one for each layer) of shape
                :obj:(batch_size, num_heads, sequence_length, sequence_length).

                Attentions weights after the attention softmax, used to compute the weighted average
                in the self-attention heads.
        """

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            image_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (lm_loss,) + outputs

        return outputs


class ReasoningClassification(nn.Module):
    def __init__(self, txt_dim, image_dim, inner_dim):
        super().__init__()
        self._txt_dim = txt_dim
        self._image_dim = image_dim
        self.txt_proj = nn.Linear(txt_dim, inner_dim)
        self.image_proj = nn.Linear(image_dim, inner_dim)
        self.out_proj = nn.Linear(2 * inner_dim, 2)
        self.act_fct = nn.Tanh()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, txt, image, label):
        txt_x = self.txt_proj(txt.view(-1, self._txt_dim))
        txt_x = self.act_fct(txt_x)
        image_x = self.image_proj(image.view(-1, self._image_dim))
        image_x = self.act_fct(image_x)
        x = self.out_proj(torch.cat((image_x, txt_x), dim=1))
        loss = self.loss_fct(x.view(-1, 2), label.view(-1))
        return loss
