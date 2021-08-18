import torch
from transformers import BartTokenizer
from src.utils import TaskType


class ConditionTokenizer:
    """
    tokenizer for image features, event and task type
    this is NOT inherent from transformers Tokenizer
    """

    def __init__(
            self,
            pretrained_model_name='facebook/bart-large',
            begin_img="<img>",
            end_img="</img>",
            begin_event="<event>",
            end_event="</event>",
            before="<before>",
            intent="<intent>",
            after="<after>",
            caption="<caption>",
            img_feat='<img_feat>',
            begin_mlm="<mlm>",
            end_mlm="</mlm>",
            cls_token="<cls>",
            token1="<token1>",
            token2="<token2>",
            token3="<token3>",
            region_caption="<region_caption>"
    ):
        self._base_tokenizer = BartTokenizer.from_pretrained(
            pretrained_model_name,
        )

        self.additional_special_tokens = [
            begin_img,
            end_img,
            begin_event,
            end_event,
            before,
            intent,
            after,
            caption,
            img_feat,
            begin_mlm,
            end_mlm,
            cls_token,
            token1,
            token2,
            token3,
            region_caption
        ]

        self._base_tokenizer.add_special_tokens(
            {'additional_special_tokens': self.additional_special_tokens}
        )

        self.begin_img = begin_img
        self.end_img = end_img
        self.begin_event = begin_event
        self.end_event = end_event
        self.before = before
        self.intent = intent
        self.after = after
        self.img_feat = img_feat
        self.caption = caption
        self.begin_mlm = begin_mlm
        self.end_mlm = end_mlm
        self.cls_token = cls_token
        self.region_caption = region_caption

        self.begin_img_id = self.convert_tokens_to_ids(begin_img)
        self.end_img_id = self.convert_tokens_to_ids(end_img)
        self.begin_event_id = self.convert_tokens_to_ids(begin_event)
        self.end_event_id = self.convert_tokens_to_ids(end_event)
        self.before_id = self.convert_tokens_to_ids(before)
        self.intent_id = self.convert_tokens_to_ids(intent)
        self.after_id = self.convert_tokens_to_ids(after)
        self.img_feat_id = self.convert_tokens_to_ids(img_feat)
        self.caption_id = self.convert_tokens_to_ids(caption)
        self.begin_mlm_id = self.convert_tokens_to_ids(begin_mlm)
        self.end_mlm_id = self.convert_tokens_to_ids(end_mlm)
        self.cls_token_id = self.convert_tokens_to_ids(cls_token)
        self.region_caption_id = self.convert_tokens_to_ids(region_caption)

        self.vocab_size = self._base_tokenizer.vocab_size
        self.bos_token = self._base_tokenizer.bos_token
        self.bos_token_id = self._base_tokenizer.bos_token_id
        self.eos_token = self._base_tokenizer.eos_token
        self.eos_token_id = self._base_tokenizer.eos_token_id
        self.pad_token = self._base_tokenizer.pad_token
        self.pad_token_id = self._base_tokenizer.pad_token_id
        self.unk_token = self._base_tokenizer.unk_token
        self.unk_token_id = self._base_tokenizer.unk_token_id

    def encode(self, *args, **kwargs):
        return self._base_tokenizer(*args, **kwargs)

    def encode_condition(self, task_type, img_num=None, event=None, mlm=None):
        """
        tokenize text, image features and event
        the output format (after decoded back):
        task_type [<img> <img_feat> ... <img_feat> </img>] [<event> EVENT </event>] [<mlm> MLM </mlm>]

        :param task_type: str or list[str]
        :param img_num: int or list[int], the number of image features
        :param event: str or list[str], event descriptions
        :param mlm: str or list[str], sentence for masked language modeling
        :return: dict {str: Tensor}, {
                "input_ids": ...,
                "attention_mask": ...,
                "event_mask": ...,          only exist if event is given. 1 for the position with event tokens
                "mlm_mask": ...,            only exist if mlm is given. 1 for the position with mlm tokens
                "img_mask":...,             only exist if img_num is given. 1 for the position with img tokens
            }
        """
        text = []

        # build task types, a list of
        # <intent>, <before> or <after>
        if not isinstance(task_type, list):
            task_type = [task_type]

        for value in task_type:
            if value == TaskType.INTENT:
                text.append(self.intent)
            elif value == TaskType.BEFORE:
                text.append(self.before)
            elif value == TaskType.AFTER:
                text.append(self.after)
            elif value == TaskType.CAPTION:
                text.append(self.caption)
            elif value == TaskType.REGION_CAPTION:
                text.append(self.region_caption)
            else:
                raise ValueError('Unexpected task type "{}"'.format(value))

        # build image features
        # <img> <img_feat> ... <img_feat> </img>
        if img_num is not None:
            if not isinstance(img_num, list):
                img_num = [img_num]

            for index, value in enumerate(img_num):
                text[index] += self.begin_img + self.img_feat * value + self.end_img

        # build event
        # <event> EVENT </event>
        if event is not None:
            if not isinstance(event, list):
                event = [event]

            for index, value in enumerate(event):
                text[index] += self.begin_event + value + self.end_event

        # build mlm
        # <mlm> MLM </mlm>
        if mlm is not None:
            if not isinstance(mlm, list):
                mlm = [mlm]

            for index, value in enumerate(mlm):
                text[index] += self.begin_mlm + value + self.end_mlm

        encoded = self.encode(
            text,
            add_special_tokens=False,
            return_tensors='pt',
            padding=True
        )

        # build event mask
        if event is not None:
            event_mask = torch.zeros(encoded['input_ids'].size(), dtype=torch.bool)
            for index, value in enumerate(encoded['input_ids']):
                start = (value == self.begin_event_id).nonzero(as_tuple=True)[0]
                end = (value == self.end_event_id).nonzero(as_tuple=True)[0]
                event_mask[index, start + 1: end] = True
            encoded['event_mask'] = event_mask

        # build mlm mask
        if mlm is not None:
            mlm_mask = torch.zeros(encoded['input_ids'].size(), dtype=torch.bool)
            for index, value in enumerate(encoded['input_ids']):
                start = (value == self.begin_mlm_id).nonzero(as_tuple=True)[0]
                end = (value == self.end_mlm_id).nonzero(as_tuple=True)[0]
                mlm_mask[index, start + 1: end] = True
            encoded['mlm_mask'] = mlm_mask

        # build img mask
        if img_num is not None:
            encoded['img_mask'] = encoded['input_ids'] == self.img_feat_id

        return encoded

    def encode_label(self, label, img_num=None):
        text = []

        # build text label
        # <s> LABEL </s>
        if not isinstance(label, list):
            label = [label]

        for value in label:
            text.append(self.bos_token + value + self.eos_token)

        # build image features
        # <img> <img_feat> ... <img_feat> </img>
        if img_num is not None:
            if not isinstance(img_num, list):
                img_num = [img_num]

            for index, value in enumerate(img_num):
                text[index] = self.begin_img + self.img_feat * value + self.end_img + text[index]

        encoded_label = self.encode(
            text,
            add_special_tokens=False,
            return_tensors='pt',
            padding=True
        )

        input_ids = encoded_label['input_ids']
        attention_mask = encoded_label['attention_mask']

        output_shape = input_ids[:, 1:].shape
        labels = torch.empty(output_shape, dtype=torch.long)
        decoder_input_ids = torch.empty(output_shape, dtype=torch.long)
        decoder_attention_mask = torch.empty(output_shape, dtype=torch.long)

        # remove <s> from labels, remove </s> from decoder_input_ids
        # remove the element in attention_mask at the same position as </s> in decoder_input_ids
        for i in range(labels.size(0)):
            labels[i] = input_ids[i][input_ids[i] != self.bos_token_id]
            decoder_input_ids[i] = input_ids[i][input_ids[i] != self.eos_token_id]
            decoder_attention_mask[i] = attention_mask[i][input_ids[i] != self.eos_token_id]

        output = {
            'labels': labels,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask
        }

        # build img mask
        if img_num is not None:
            output['label_img_mask'] = labels == self.img_feat_id
            output['decoder_input_img_mask'] = decoder_input_ids == self.img_feat_id

        return output

    def decode(self, token_ids, skip_special_tokens=False):
        return self._base_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def convert_tokens_to_ids(self, tokens):
        return self._base_tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self._base_tokenizer.convert_ids_to_tokens(ids)

    def get_base_tokenizer(self):
        return self._base_tokenizer

    def __len__(self):
        return len(self._base_tokenizer)
