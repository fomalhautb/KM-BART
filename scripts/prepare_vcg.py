import argparse
import json
import os
import pickle
import warnings
from datetime import datetime

import cv2
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm

from scripts.utils import print_segment_line
from .feature_extractor import FeatureExtractor


def get_img_id(annot):
    img_id = annot['img_fn']
    img_id = os.path.basename(img_id)
    img_id = img_id[:img_id.rfind('.')]
    return img_id


def get_image_data(annot, data_dir, extractor):
    img_dir = annot['img_fn']
    metadata_dir = annot['metadata_fn']

    im = cv2.imread(os.path.join(data_dir, img_dir))
    metadata = json.load(open(os.path.join(data_dir, metadata_dir)))

    boxes = np.array(metadata['boxes'])[:, :4]
    h = metadata['height']
    w = metadata['width']
    boxes = np.row_stack((np.array([0, 0, w, h]), boxes))

    features = extractor.extract_feature(im, boxes)

    return {
        'image_features': features['features'],
        'mrm_labels': features['scores'],
        'boxes': features['boxes']
    }


def get_text_data(annot, index):
    data = []
    event = annot['event']
    img_id = get_img_id(annot)
    base_entry = {'event': event, 'img_id': img_id, 'img_fn': annot['img_fn'], 'index': index}

    if annot['split'] == 'test':
        data.append(base_entry)
    else:
        for intent in annot['intent']:
            data.append({**base_entry, 'task_type': 'intent', 'labels': intent})
        for before in annot['before']:
            data.append({**base_entry, 'task_type': 'before', 'labels': before})
        for after in annot['after']:
            data.append({**base_entry, 'task_type': 'after', 'labels': after})

    return data


def get_eval_data(annot, index):
    data = []
    event = annot['event']
    img_id = get_img_id(annot)
    base_entry = {'event': event, 'img_id': img_id, 'img_fn': annot['img_fn'], 'index': index}

    if annot['split'] == 'test':
        data.append(base_entry)
    else:
        data.append({**base_entry, 'task_type': 'intent'})
        data.append({**base_entry, 'task_type': 'after'})
        data.append({**base_entry, 'task_type': 'before'})

    return data


def get_reference_data(annot):
    return [{
        'intent': annot.get('intent'),
        'before': annot.get('before'),
        'after': annot.get('after')
    }]


def main(rank, data, data_dir, split, args):
    extractor = FeatureExtractor(args.config, rank)
    local_data = data[rank::args.gpu_num]
    start_time = datetime.now()

    for i, entry in enumerate(local_data):
        data = get_image_data(annot=entry, data_dir=args.data_dir, extractor=extractor)
        pickle.dump(data, open(os.path.join(args.output_dir, split, get_img_id(entry) + '.pkl'), 'wb'))

        print('GPU{}, {}/{}, ETA: {}'.format(
            rank,
            i,
            len(local_data),
            str((len(local_data) - (i + 1)) / (i + 1) * (datetime.now() - start_time))
        ), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='VCR dataset directory. None for not generating image features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--annot_dir', type=str, required=True,
                        help='VCG annotation directory'
                             'with "val_annots.json", "train_annots.json" and "test_annots.json"')
    parser.add_argument('--config', type=str, default=None,
                        help='path extractor config')
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='number of GPUs in total')
    args = parser.parse_args()

    if args.config is None:
        args.config = 'config/extract_config.yaml'

    # load annotations
    train_annots = json.load(open(os.path.join(args.annot_dir, 'train_annots.json')))
    val_annots = json.load(open(os.path.join(args.annot_dir, 'val_annots.json')))
    test_annots = json.load(open(os.path.join(args.annot_dir, 'test_annots.json')))

    split_dict = {'train': train_annots, 'val': val_annots, 'test': test_annots}

    # make directory for splits
    for split in split_dict.keys():
        path = os.path.join(args.output_dir, split)
        if not os.path.isdir(path):
            os.mkdir(path)

    # generate and save training data (event, task_type, etc.)
    # [{
    #       'event': sentence, 'img_id': image_id, 'img_fn': image_path,
    #       'index': event_index, 'task_type': task_type, 'labels': sentence
    #  }, ...]
    print_segment_line('processing training data')
    for split, annots in split_dict.items():
        data = []
        for index, annot in enumerate(tqdm(annots)):
            data += get_text_data(annot=annot, index=index)
        json.dump(data, open(os.path.join(args.output_dir, split + '.json'), 'w'))

    # generate and save evaluation data (event, task_type, etc.)
    # [{
    #       'event': sentence, 'img_id': id, 'img_fn': image_path,
    #       'index': event_index, 'task_type': task_type
    #  }, ...]
    print_segment_line('processing evaluation data')
    for split, annots in split_dict.items():
        data = []
        for index, annot in enumerate(tqdm(annots)):
            data += get_eval_data(annot=annot, index=index)
        json.dump(data, open(os.path.join(args.output_dir, split + '_eval.json'), 'w'))

    # generate and save reference data
    # [{
    #       'intent': [sentence1, sentence2, ...],
    #       'before': [sentence3, sentence4, ...],
    #       'after': [sentence5, sentence6, ...]
    #  }, ...]
    print_segment_line('processing reference data')
    for split, annots in split_dict.items():
        if split != 'test':
            data = []
            for index, annot in enumerate(tqdm(annots)):
                data += get_reference_data(annot=annot)
            json.dump(data, open(os.path.join(args.output_dir, split + '_ref.json'), 'w'))

    # generate and save image features
    # {'image_features': image_features}
    # first feature is of the entire feature, the remaining are object features
    if args.data_dir is not None:
        for split, annots in split_dict.items():
            print_segment_line('extracting image features for {} set'.format(split))
            mp.spawn(
                main,
                args=(annots, args.data_dir, split, args),
                nprocs=args.gpu_num,
                join=True
            )
