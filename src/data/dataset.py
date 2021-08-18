import json
import os
import pickle
import cv2
import numpy as np

from torch.utils.data import Dataset

from src.utils import TaskType

"""
The dataset return format: a dictionary
{
    'task_type': ...        # TaskType
    'image_features': ...   # list[ndarray], optional
    'event': ...            # str, optional
    'labels': ...           # str, optional
    'index': ...            # int, optional, the index of reference data
    other task specific items...
}
"""


class COCODataset(Dataset):
    def __init__(self, data_dir, image_dir=None, split='train', eval_mode=False, use_image=True):
        """
        :param eval_mode: bool, if set to true, "xxx_eval.json" will be loaded, every image will only appear one time
        """
        self._use_image = use_image
        self._data_dir = data_dir
        self._image_dir = data_dir if image_dir is None else image_dir
        self._split = split

        file_name = split + ('_eval.json' if eval_mode else '.json')
        self._dataset = json.load(open(os.path.join(data_dir, file_name), 'r'))

    def __getitem__(self, index):
        raw_data = self._dataset[index]
        output = {**raw_data}

        if self._use_image:
            image_dir = os.path.join(self._image_dir, self._split, str(raw_data['img_id']) + '.pkl')
            image_data = pickle.load(open(image_dir, 'rb'))
            output['image_features'] = np.concatenate([
                image_data['image_features'],
                image_data['boxes']
            ], axis=1).astype(np.float32)

            if 'mrm_labels' in image_data:
                output['mrm_labels'] = image_data['mrm_labels']

        return output

    def __len__(self):
        return len(self._dataset)


class VCGDataset(COCODataset):
    def __init__(
            self,
            data_dir,
            image_dir=None,
            split='train',
            eval_mode=False,
            use_image=True,
            use_event=True,
            pretrain=False,
    ):
        super(VCGDataset, self).__init__(
            data_dir=data_dir,
            image_dir=image_dir,
            split=split,
            eval_mode=eval_mode,
            use_image=use_image
        )
        self._use_event = use_event
        self._pretrain = pretrain

    def __getitem__(self, item):
        output = super(VCGDataset, self).__getitem__(item)

        if not self._use_event:
            output['event'] = output['event'].split()[0]  # only show the target person
        if self._pretrain:
            output['labels'] = output['event']
            del output['event']
            output['task_type'] = TaskType.CAPTION

        return output


class SBUDataset(COCODataset):
    def __init__(self, data_dir, image_dir=None, split='train', use_image=True):
        super(SBUDataset, self).__init__(
            data_dir=data_dir,
            image_dir=image_dir,
            split=split,
            eval_mode=False,
            use_image=use_image
        )

    def __getitem__(self, item):
        output = super(SBUDataset, self).__getitem__(item)
        output['task_type'] = TaskType.CAPTION
        output['labels'] = output['labels'].strip()
        return output


class CCDataset(SBUDataset):
    pass


class VGDataset(Dataset):
    def __init__(self, data_dir, image_dir=None, split='train'):
        self._data_dir = data_dir
        self._image_dir = data_dir if image_dir is None else image_dir
        self._split = split

        self._dataset = json.load(open(os.path.join(data_dir, split + '.json'), 'r'))
        self._region_dataset = json.load(open(os.path.join(data_dir, split + '_region.json'), 'r'))

    def __len__(self):
        return len(self._region_dataset)

    def __getitem__(self, index):
        region_data = self._region_dataset[index]
        img_id = region_data['img_id']
        region_id = region_data['region_id']
        raw_data = self._dataset[str(img_id)]
        output = {**raw_data}

        image_dir = os.path.join(self._image_dir, self._split, str(raw_data['img_id']) + '.pkl')
        image_data = pickle.load(open(image_dir, 'rb'))

        region_index = image_data['region_ids'].index(region_id)
        region_feature = np.concatenate([
            image_data['region_features'][region_index],
            image_data['region_boxes'][region_index]
        ], axis=0)

        image_feature = np.concatenate([
            image_data['image_feature'],
            image_data['image_box']
        ], axis=0)

        object_features = np.concatenate([
            image_data['object_features'],
            image_data['object_boxes']
        ], axis=1)

        output['image_features'] = np.concatenate([
            image_feature[np.newaxis, :],
            object_features,
            region_feature[np.newaxis, :]
        ], axis=0)

        output['mrm_labels'] = np.concatenate([
            image_data['image_score'][np.newaxis, :],
            image_data['object_scores'],
            image_data['region_scores'][region_index: region_index+1]
        ], axis=0)

        output['object_ids'] = image_data['object_ids']
        output['task_type'] = TaskType.REGION_CAPTION
        output['labels'] = region_data['description']

        return output


class ReasonDataset(Dataset):
    def __init__(self, data_dir, image_dir=None, split='train', eval_mode=False, use_image=True, use_event=True):
        """
        :param eval_mode: bool, if set to true, "xxx_eval.json" will be loaded, every image will only appear one time
        """
        self._use_image = use_image
        self._use_event = use_event
        self._data_dir = data_dir
        self._image_dir = data_dir if image_dir is None else image_dir
        self._split = split

        file_name = 'reason_' + split + ('_eval.json' if eval_mode else '.json')
        self._dataset = json.load(open(os.path.join(data_dir, file_name), 'r'))

    def __getitem__(self, index):
        raw_data = self._dataset[index]
        output = {**raw_data}
        
        if not self._use_event:
            output['event'] = ''
            
        if self._use_image:
            try:
                image_dir = os.path.join(self._image_dir, self._split, str(raw_data['img_id']) + '.pkl')
                image_data = pickle.load(open(image_dir, 'rb'))
            except FileNotFoundError:
                return None

            output['image_features'] = np.concatenate([
                image_data['image_features'],
                image_data['boxes']
            ], axis=1).astype(np.float32)

            if 'mrm_labels' in image_data:
                output['mrm_labels'] = image_data['mrm_labels']

        output['dataset_index'] = index

        return output

    def get_raw_data(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


