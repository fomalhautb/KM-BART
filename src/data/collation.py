import warnings

import numpy as np
import torch

from src.utils import TaskType


class Collator:
    """
    The collator for all types of dataset.
    Remember to add the corresponding collation code after adding a new type of task.
    """

    def __init__(
            self,
            tokenizer,
            has_label=True,
            mlm_enabled=False,
            mrm_enabled=False,
            rp_enabled=False,
            ap_enabled=False,
            mlm_probability=0.0,
            mrm_probability=0.0,
            event_max_len=20,
            lm_max_len=30,
            max_img_num=30,
            max_rel_count=80
    ):
        """
        :param tokenizer: ConditionTokenizer
        :param mlm_enabled: bool, if use mlm for language modeling. False for autoregressive modeling
        :param mrm_enabled: bool, if use mrm
        :param rp_enabled: bool, if use relation prediction (VG)
        :param ap_enabled: bool, if use attribute prediction (VG)
        :param mlm_probability: float, probability to mask the tokens
        :param mrm_probability: float, probability to mask the regions
        """
        self._tokenizer = tokenizer
        self._has_label = has_label
        self._mlm_enabled = mlm_enabled
        self._mrm_enabled = mrm_enabled
        self._rp_enabled = rp_enabled
        self._ap_enabled = ap_enabled
        self._mlm_probability = mlm_probability
        self._mrm_probability = mrm_probability
        self._event_max_len = event_max_len
        self._lm_max_len = lm_max_len
        self._max_img_num = max_img_num
        self._max_rel_count = max_rel_count

        if mlm_enabled and not has_label:
            raise ValueError('mlm_enabled can not be true while has_label is false. MLM need labels.')

        if ap_enabled and not has_label:
            raise ValueError('ap_enabled can not be true while has_label is false. attribute prediction need labels.')

        if rp_enabled and not has_label:
            raise ValueError('rp_enabled can not be true while has_label is false. relation prediction need labels.')

        if (rp_enabled or ap_enabled) and not mrm_enabled:
            raise ValueError('if rp/ap is enabled, mrm must also be enabled')

    def _clip_text(self, text, length):
        tokenized = self._tokenizer.get_base_tokenizer()(text, add_special_tokens=False)
        return self._tokenizer.get_base_tokenizer().decode(tokenized['input_ids'][:length])

    def __call__(self, batch):
        batch = [entry for entry in batch if entry is not None]
        if not all([x['task_type'] in TaskType.ALL_TYPES for x in batch]):
            warnings.warn('Unexpected task type in batch')

        image_features = [
            torch.from_numpy(x['image_features'][:self._max_img_num]) if 'image_features' in x else torch.empty(0) for x
            in batch
        ]

        img_num = [len(x) for x in image_features]
        label_img_num = img_num if self._mrm_enabled else None
        # take the first event_max_len words
        event = [self._clip_text(x['event'], self._event_max_len) if 'event' in x else '' for x in batch]
        task_type = [x['task_type'] for x in batch]
        # take the first lm_max_len words
        target = [self._clip_text(x['labels'], self._lm_max_len) for x in batch] if self._has_label else None
        mlm = list(target) if self._mlm_enabled else None
        for i in range(len(batch)):
            if batch[i]['task_type'] in ['before', 'after', 'intent'] and self._mlm_enabled:
                mlm[i] = event[i]
                event[i] = ''

        encoded_conditions = self._tokenizer.encode_condition(
            img_num=img_num,
            event=event,
            task_type=task_type,
            mlm=mlm
        )

        input_ids = encoded_conditions['input_ids']

        if self._mlm_enabled:
            input_ids = self._mask_tokens(inputs=input_ids, input_mask=encoded_conditions['mlm_mask'])

        output = {
            'input_ids': input_ids,
            'attention_mask': encoded_conditions['attention_mask'],
            'image_features': image_features,
            'index': [x['index'] if 'index' in x else None for x in batch],
            'task_type': [x['task_type'] for x in batch]
        }

        condition_img_mask = encoded_conditions['img_mask']

        if self._mrm_enabled:
            # mrm on input_ids
            probability_matrix = torch.full(input_ids.shape, self._mrm_probability, dtype=torch.float)
            masked_regions = torch.bernoulli(probability_matrix).bool()
            input_ids[masked_regions & condition_img_mask] = self._tokenizer.cls_token_id
            mrm_labels = []

            for i in range(len(batch)):
                # create mrm_labels
                masked_indices = masked_regions[i][condition_img_mask[i]].nonzero(as_tuple=False)
                mrm_label = torch.Tensor(batch[i]['mrm_labels'][:self._max_img_num])
                mrm_labels.append(mrm_label[masked_indices].view(-1, mrm_label.shape[1]).clone())

                if len(image_features[i]) > 0:
                    image_features[i][masked_indices] = torch.cat([
                        torch.zeros((len(masked_indices), 1, 2048), dtype=image_features[i].dtype),  # mask images
                        image_features[i][masked_indices][:, :, -4:]  # keep bbox features
                    ], axis=2)

            output['mrm_labels'] = mrm_labels

        if self._has_label:
            # encode mrm and mlm labels
            encoded_labels = self._tokenizer.encode_label(label=target, img_num=label_img_num)
            labels = encoded_labels['labels']
            decoder_input_ids = encoded_labels['decoder_input_ids']

            # mrm on labels and decoder_input_ids
            if self._mrm_enabled:
                label_img_mask = encoded_labels['label_img_mask']
                labels[label_img_mask] = input_ids[condition_img_mask]

                decoder_input_img_mask = encoded_labels['decoder_input_img_mask']
                decoder_input_ids[decoder_input_img_mask] = input_ids[condition_img_mask]

            # parse attribute prediction labels
            if self._ap_enabled:
                attribute_mask = torch.zeros(labels.size())
                attribute_labels = [[] for _ in range(len(batch))]

                for index, entry in enumerate(batch):
                    if 'object_ids' in entry:  # check if entry is from VG
                        start_pos = (labels[index] == self._tokenizer.begin_img_id).nonzero(as_tuple=True)[0][0] + 2
                        obj_dict = {o['object_id']: o for o in entry['objects']}

                        for obj_pos, obj_id in enumerate(entry['object_ids'][:self._max_img_num - 2]):
                            if 'attribute_ids' in obj_dict[obj_id]:
                                attribute_mask[index][obj_pos + start_pos] = 1
                                attribute_id = obj_dict[obj_id]['attribute_ids'][0]  # always take the first attribute
                                attribute_labels[index].append(attribute_id)

                output['attribute_labels'] = [torch.LongTensor(x) for x in attribute_labels]
                output['attribute_mask'] = attribute_mask

            if self._rp_enabled:
                relation_labels = [[] for _ in range(len(batch))]

                for index, entry in enumerate(batch):
                    if 'object_ids' in entry:  # check if entry is from VG
                        rel_count = 0
                        start_pos = (labels[index] == self._tokenizer.begin_img_id).nonzero(as_tuple=True)[0][
                                        0].item() + 2
                        obj_pos_dict = {j: start_pos + i for i, j in
                                        enumerate(entry['object_ids'][:self._max_img_num - 2])}
                        for rel in entry['relations']:
                            if rel['object_id'] in obj_pos_dict and rel['subject_id'] in obj_pos_dict:
                                relation_labels[index].append({
                                    'object_index': obj_pos_dict[rel['object_id']],
                                    'subject_index': obj_pos_dict[rel['subject_id']],
                                    'label': rel['predicate_id']
                                })

                                # only take the first max_rel_count relations
                                rel_count += 1
                                if rel_count >= self._max_rel_count:
                                    break

                output['relation_labels'] = relation_labels

            labels[(labels == self._tokenizer.pad_token_id) |
                   (labels == self._tokenizer.begin_img_id) |
                   (labels == self._tokenizer.end_img_id) |
                   (labels == self._tokenizer.img_feat_id)] = -100

            output['labels'] = labels
            output['decoder_input_ids'] = decoder_input_ids
            output['decoder_attention_mask'] = encoded_labels['decoder_attention_mask']

            if self._mrm_enabled:
                output['mrm_mask'] = labels == self._tokenizer.cls_token_id

        if 'question_id' in batch[0]:
            output['question_id'] = [x['question_id'] for x in batch]

        if 'dataset_index' in batch[0]:
            output['dataset_index'] = [x['dataset_index'] if 'dataset_index' in x else None for x in batch]

        if self._has_label:
            output['raw_labels'] = [x['labels'] for x in batch]

        return output

    # based on transformers.data.data_collator.DataCollatorForLanguageModeling
    def _mask_tokens(self, inputs, input_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        :param inputs: torch.LongTensor, batch data
        :param input_mask: torch.Tensor, mask for the batch, False for the position with 0% probability to be masked
        """

        labels = inputs.clone()
        tokenizer = self._tokenizer.get_base_tokenizer()

        # We sample a few tokens in each sequence for masked-LM training
        probability_matrix = torch.full(labels.shape, self._mlm_probability, dtype=torch.float)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                               for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if tokenizer.pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced & input_mask] = tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random & input_mask] = random_words[indices_random & input_mask]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs


class AtomicCollator:
    def __init__(self, tokenizer, txt_backbone, image_seq_length, txt_seq_length, shuffle_ratio):
        self._tokenizer = tokenizer
        self._txt_backbone = txt_backbone
        self._image_seq_length = image_seq_length
        self._txt_seq_length = txt_seq_length
        self._shuffle_ratio = shuffle_ratio

    def __call__(self, batch):
        output = {}
        event = np.array([x['event'] if 'event' in x else '' for x in batch])
        label = np.ones(len(batch), dtype=np.long)
        new_order = np.arange(len(batch))
        np.random.shuffle(new_order)
        probability_matrix = np.random.random(len(batch))
        masked_regions = probability_matrix > self._shuffle_ratio
        event[masked_regions] = event[new_order[masked_regions]]
        label[masked_regions] = 0
        output['label'] = torch.from_numpy(label)

        txt_input = self._tokenizer(
            event.tolist(),
            max_length=self._txt_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        txt_rep = self._txt_backbone(
            input_ids=txt_input['input_ids'],
            attention_mask=txt_input['attention_mask'],
            token_type_ids=txt_input['token_type_ids'],
        )
        output['text'] = txt_rep[0]
        image_features = [
            x['image_features'][:self._image_seq_length] if 'image_features' in x else torch.empty(0) for x in batch
        ]

        image_rep = np.array([
            np.concatenate((feat, np.zeros((self._image_seq_length - len(feat), 2052))), axis=0) for feat in
            image_features
        ], dtype=np.float32)

        output['image'] = torch.from_numpy(image_rep)

        return output
