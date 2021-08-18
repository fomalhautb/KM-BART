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

# from .feature_extractor import FeatureExtractor
from .atomic_generator import AtomicGenerator
from .utils import print_segment_line


def extract_data(captions, instances):
    data = {}

    # extract image metadata
    for img in tqdm(captions['images']):
        img_id = img['id']
        data[img_id] = {
            'img_id': img_id,
            'img_fn': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # extract captions
    for cap in tqdm(captions['annotations']):
        img_id = cap['image_id']

        if 'caption' not in data[img_id]:
            data[img_id]['caption'] = []

        data[img_id]['caption'].append(cap['caption'])

    # extract bounding boxes
    for ins in tqdm(instances['annotations']):
        img_id = ins['image_id']

        if 'boxes' not in data[img_id]:
            data[img_id]['boxes'] = []
        boxes = ins['bbox']
        boxes[2] += boxes[0]
        boxes[3] += boxes[1]
        data[img_id]['boxes'].append(boxes)

    # remove incomplete data entries
    for key in tqdm(list(data.keys())):
        if 'caption' not in data[key]:
            data[key]['caption'] = ''
    # data_t = dict(list(data.items())[:100])
    return data


def get_text_data(entry, index, atomic_generator):
    data = []
    base_entry = {
        'img_id': str(entry['img_id']),
        'img_fn': entry['img_fn'],
        'index': index,
    }

    ref = []
    for caption in entry['caption']:
        res = atomic_generator.get_reason(caption)
        for k in ['before', 'after', 'intent']:
            for ans in res[k]:
                data.append({**base_entry, 'event': caption, 'task_type': k, 'labels': ans})
        ref.append(res)

    return data, ref


def get_eval_data(entry, index, ref_ans):
    data = []
    base_entry = {
        'img_id': str(entry['img_id']),
        'img_fn': entry['img_fn'],
        'index': index,
    }

    for i, caption in enumerate(entry['caption']):
        if len(ref_ans[i]['before']) != 0:
            data.append({**base_entry, 'event': caption, 'task_type': 'before'})
        if len(ref_ans[i]['after']) != 0:
            data.append({**base_entry, 'event': caption, 'task_type': 'after'})
        if len(ref_ans[i]['intent']) != 0:
            data.append({**base_entry, 'event': caption, 'task_type': 'intent'})
    return data


def get_reference_data(entry, ref_ans):
    data = []
    for i, _ in enumerate(entry['caption']):
        # if len(ref_ans[i]['before']) == 0 or len(ref_ans[i]['after']) == 0 or len(ref_ans[i]['intent']) == 0:
        #     continue
        data.append(ref_ans[i])
    return data


def process_text(rank, args, split, split_dict):
    generator = AtomicGenerator(args, rank)
    local_data = split_dict[rank::args.gpu_num]
    start_time = datetime.now()

    text_data = []
    eval_data = []
    ref_data = []
    for i, entry in enumerate(local_data):
        text_data_tmp, ref_ans = get_text_data(entry=entry, index=i, atomic_generator=generator)
        text_data += text_data_tmp
        eval_data += get_eval_data(entry=entry, index=i, ref_ans=ref_ans)
        ref_data += get_reference_data(entry=entry, ref_ans=ref_ans)
        print('GPU{}, {}/{}, ETA: {}'.format(
            rank,
            i,
            len(local_data),
            str((len(local_data) - (i + 1)) / (i + 1) * (datetime.now() - start_time))
        ), flush=True)
    json.dump(text_data, open(os.path.join(args.output_dir, split + str(rank) + '.json'), 'w'))
    json.dump(eval_data, open(os.path.join(args.output_dir, split + str(rank) + '_eval.json'), 'w'))
    json.dump(ref_data, open(os.path.join(args.output_dir, split + str(rank) + '_ref.json'), 'w'))


def get_image_data(entry, data_dir, extractor):
    img_dir = entry['img_fn']

    im = cv2.imread(os.path.join(data_dir, img_dir))

    h = entry['height']
    w = entry['width']
    boxes = np.array([0, 0, w, h])

    if 'boxes' in entry:
        boxes = np.row_stack((np.array(entry['boxes']), boxes))
    else:
        boxes = np.row_stack((boxes,))

    features = extractor.extract_feature(im, boxes)

    return {
        'image_features': features['features'],
        'mrm_labels': features['scores'],
        'boxes': features['boxes']
    }


def main(rank, data, data_dir, split, args):
    extractor = FeatureExtractor(args.config, rank)
    local_data = data[rank::args.gpu_num]
    start_time = datetime.now()

    for i, entry in enumerate(local_data):
        image_data = get_image_data(entry=entry, data_dir=data_dir, extractor=extractor)
        pickle.dump(image_data, open(os.path.join(args.output_dir, split, str(entry['img_id']) + '.pkl'), 'wb'))
        print('GPU{}, {}/{}, ETA: {}'.format(
            rank,
            i,
            len(local_data),
            str((len(local_data) - (i + 1)) / (i + 1) * (datetime.now() - start_time))
        ), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=None,
                        help='path for training images (train2014). None for not generating image features')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='path for validation images (val2014). None for not generating image features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--annot_dir', type=str, required=True,
                        help='VCG annotation directory with "captions_train2014.json", '
                             '"captions_val2014.json", "instances_train2014.json" and "instances_val2014.json"')
    parser.add_argument('--config', type=str, default='config/extract_config.yaml',
                        help='path extractor config')
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='number of GPUs in total')
    parser.add_argument("--model_file", type=str,
                        default="comet-commonsense/model/pretrained_models/atomic_pretrained_model.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="beam-1")

    args = parser.parse_args()

    print_segment_line('extracting training annotations')

    train_data = extract_data(
        captions=json.load(open(os.path.join(args.annot_dir, 'captions_train2014.json'), 'r')),
        instances=json.load(open(os.path.join(args.annot_dir, 'instances_train2014.json'), 'r'))
    )

    print_segment_line('extracting validation annotations')

    val_data = extract_data(
        captions=json.load(open(os.path.join(args.annot_dir, 'captions_val2014.json'), 'r')),
        instances=json.load(open(os.path.join(args.annot_dir, 'instances_val2014.json'), 'r'))
    )

    split_dict = {'train': (train_data, args.train_dir), 'val': (val_data, args.val_dir)}

    # make directory for splits
    for split in split_dict.keys():
        path = os.path.join(args.output_dir, split)
        if not os.path.isdir(path):
            os.mkdir(path)

    print_segment_line('generating textual and reference data')

    # generate and save training data, evaluation data and reference data
    for split, (data, _) in split_dict.items():
        print_segment_line('generate comet reason for {} set'.format(split))
        mp.spawn(
            process_text,
            args=(args, split, list(data.values())),
            nprocs=args.gpu_num,
            join=True
        )
    for split in split_dict.keys():
        text_data = []
        eval_data = []
        ref_data = []
        for rank in range(args.gpu_num):
            text_data += json.load(open(os.path.join(args.output_dir, split + str(rank) + '.json'), 'r'))
            eval_data += json.load(open(os.path.join(args.output_dir, split + str(rank) + '_eval.json'), 'r'))
            ref_data += json.load(open(os.path.join(args.output_dir, split + str(rank) + '_ref.json'), 'r'))
            os.remove(os.path.join(args.output_dir, split + str(rank) + '.json'))
            os.remove(os.path.join(args.output_dir, split + str(rank) + '_eval.json'))
            os.remove(os.path.join(args.output_dir, split + str(rank) + '_ref.json'))
        json.dump(text_data, open(os.path.join(args.output_dir, split + '.json'), 'w'))
        json.dump(eval_data, open(os.path.join(args.output_dir, split + '_eval.json'), 'w'))
        json.dump(ref_data, open(os.path.join(args.output_dir, split + '_ref.json'), 'w'))

    # extract and save image features
    for split, (data, data_dir) in split_dict.items():
        if data_dir is not None:
            print_segment_line('extracting image features for {} set'.format(split))
            mp.spawn(
                main,
                args=(list(data.values()), data_dir, split, args),
                nprocs=args.gpu_num,
                join=True
            )
