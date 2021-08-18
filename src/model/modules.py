import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bart import (
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    invert_mask,
    EncoderLayer,
    LayerNorm,
)

from src.model.config import MultiModalBartConfig


class ImageEmbedding(nn.Module):
    def __init__(self, image_dim, final_dim):
        super(ImageEmbedding, self).__init__()
        self.linear = nn.Linear(image_dim, final_dim)

    def forward(self, image_features):
        img_len = list(map(len, image_features))
        non_empty_features = list(filter(lambda x: len(x) > 0, image_features))

        embedded = None
        if len(non_empty_features) > 0:
            img_tensor = torch.cat(non_empty_features, dim=0)
            embedded = self.linear(img_tensor)

        output = []
        index = 0
        for l in img_len:
            if l > 0:
                output.append(embedded[index: index + l])
            else:
                output.append(torch.empty(0))
            index += l
        return output


# This is copied from transformers.BartEncoder
# The modifications are:
# - added embed_images layer
# - added _embed_multi_modal function
# - added image_features in forward
class MultiModalBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """

    def __init__(self, config: MultiModalBartConfig, embed_tokens):
        super().__init__()

        self.img_feat_id = config.img_feat_id
        self.cls_token_id = config.cls_token_id

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        self.embed_images = ImageEmbedding(config.image_feature_size, embed_dim)
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

    def _embed_multi_modal(self, input_ids, image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        mask = (input_ids == self.img_feat_id) | (input_ids == self.cls_token_id)
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)

        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] = value

        return embedded

    def forward(
            self,
            input_ids,
            image_features,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False
    ):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """

        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(input_ids, image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return x, encoder_states, all_attentions
