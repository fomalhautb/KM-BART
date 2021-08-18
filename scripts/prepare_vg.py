import argparse
import json
import os
import pickle
import warnings
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm

from .feature_extractor import FeatureExtractor
from .utils import print_segment_line


def extract_relation_data(image_ids, attribute_data, relation_data, object_data):
    data = {}
    for i in image_ids:
        data[i] = {'img_id': i, 'regions': [], 'objects': {}, 'relations': []}

    for entry in tqdm(region_data):
        if entry['id'] in data:
            data[entry['id']]['regions'] = [
                {
                    'region_id': x['region_id'],
                    'description': x['phrase'],
                    'x': x['x'],
                    'y': x['y'],
                    'h': x['height'],
                    'w': x['width']
                } for x in entry['regions']
            ]

    for entry in tqdm(object_data):
        if entry['image_id'] in data:
            data[entry['image_id']]['objects'] = {
                x['object_id']: {
                    'object_id': x['object_id'],
                    'x': x['x'],
                    'y': x['y'],
                    'h': x['h'],
                    'w': x['w']
                } for x in entry['objects']
            }

    for entry in tqdm(attribute_data):
        if entry['image_id'] in data and 'attributes' in entry:
            for x in entry['attributes']:
                object_entry = data[entry['image_id']]['objects']
                if x['object_id'] in object_entry and 'attributes' in x:
                    object_entry[x['object_id']]['attributes'] = [y.lower().strip() for y in x['attributes']]

    for entry in tqdm(relation_data):
        if entry['image_id'] in data:
            data[entry['image_id']]['relations'] = [
                {
                    'object_id': x['object']['object_id'],
                    'subject_id': x['subject']['object_id'],
                    'predicate': x['predicate'].lower().strip()
                } for x in entry['relationships']
            ] if len(entry['relationships']) > 0 else []

    for entry in tqdm(data.values()):
        entry['objects'] = list(entry['objects'].values())

    return data


def extract_region_data(data, region_data):
    output = []
    for entry in tqdm(region_data):
        if entry['id'] in data:
            output += [
                {
                    'img_id': entry['id'],
                    'region_id': x['region_id'],
                    'description': x['phrase']
                } for x in entry['regions']
            ]

    return output


def get_image_dir(image_id, image_dirs):
    for image_dir in image_dirs:
        path = os.path.join(image_dir, str(image_id) + '.jpg')
        if os.path.isfile(path):
            return path
    raise FileNotFoundError('cannot find {}.jpg'.format(str(image_id)))


def get_image_data(entry, image_dirs, extractor):
    image_id = entry['img_id']
    image_dir = get_image_dir(image_id, image_dirs)
    im = cv2.imread(image_dir)
    regions = entry['regions']
    objects = entry['objects']

    boxes = np.array(
        [[r['x'], r['y'] - r['h'], r['x'] + r['w'], r['y']] for r in regions] +
        [[o['x'], o['y'] - o['h'], o['x'] + o['w'], o['y']] for o in objects] +
        [[0, 0, im.shape[1], im.shape[0]]]
    )

    features = extractor.extract_feature(im, boxes)
    image_features = features['features']
    scores = features['scores']
    boxes = features['boxes']

    return {
        'region_features': image_features[:len(regions)],
        'region_scores': scores[:len(regions)],
        'region_boxes': boxes[:len(regions)],
        'region_ids': [r['region_id'] for r in regions],
        'object_features': image_features[len(regions):-1],
        'object_scores': scores[len(regions):-1],
        'object_boxes': boxes[len(regions):-1],
        'object_ids': [o['object_id'] for o in objects],
        'image_feature': image_features[-1],
        'image_score': scores[-1],
        'image_box': boxes[-1]
    }


def main(rank, data, split, args):
    extractor = FeatureExtractor(args.config, rank)
    local_data = data[rank::args.gpu_num]
    start_time = datetime.now()

    for i, entry in enumerate(local_data):
        output = get_image_data(entry=entry, image_dirs=args.image_dir, extractor=extractor)
        pickle.dump(output, open(os.path.join(args.output_dir, split, str(entry['img_id']) + '.pkl'), 'wb'))
        print('GPU{}, {}/{}, ETA: {}'.format(
            rank,
            i,
            len(local_data),
            str((len(local_data) - (i + 1)) / (i + 1) * (datetime.now() - start_time))
        ), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Extract the ROI pooled features from images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--annot_dir', type=str, required=True,
                        help='annotation directory with "attributes.json", "objects.json"'
                             '"image_data.json", "region_descriptions.json" and "relationships.json"')
    parser.add_argument('--image_dir', nargs='*', type=str,
                        help='image directory. None for not extract image features. Can have multiple values, '
                             'the program will search through for the images in each of these folders.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='proportion of the training set, '
                             '(1-train_ratio) will be the proportion of the validation set')
    parser.add_argument('--config', type=str, default=None,
                        help='path extractor config')
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='number of GPUs in total')
    parser.add_argument('--num_relations', type=int, default=128,
                        help='the top num_relations frequently occur relations will be taken'
                             'an unknown id will be added to the end (with the id=num_relations)')
    parser.add_argument('--num_attributes', type=int, default=128,
                        help='the top num_attributes frequently occur attributes will be taken'
                             'an unknown id will be added to the end (with the id=num_attributes)')
    args = parser.parse_args()

    if args.config is None:
        args.config = 'config/extract_config.yaml'

    print_segment_line('loading data')

    print('Loading image_data', flush=True)
    image_data = json.load(open(os.path.join(args.annot_dir, 'image_data.json')))

    print('Loading attributes', flush=True)
    attribute_data = json.load(open(os.path.join(args.annot_dir, 'attributes.json')))

    print('Loading region_descriptions', flush=True)
    region_data = json.load(open(os.path.join(args.annot_dir, 'region_descriptions.json')))

    print('loading relationships', flush=True)
    relation_data = json.load(open(os.path.join(args.annot_dir, 'relationships.json')))

    print('loading objects', flush=True)
    object_data = json.load(open(os.path.join(args.annot_dir, 'objects.json')))

    image_ids = [x['image_id'] for x in image_data]
    split_index = int(len(image_ids) * args.train_ratio)
    train_image_ids = image_ids[:split_index]
    val_image_ids = image_ids[split_index:]

    print('Number of images:', len(image_ids), flush=True)
    print('Size of the training set:', len(train_image_ids), flush=True)
    print('Size of the validation set:', len(val_image_ids), flush=True)

    print_segment_line('extracting data')

    # ====================== extract data =========================
    train_data = extract_relation_data(
        image_ids=train_image_ids,
        object_data=object_data,
        relation_data=relation_data,
        attribute_data=attribute_data
    )

    val_data = extract_relation_data(
        image_ids=val_image_ids,
        object_data=object_data,
        relation_data=relation_data,
        attribute_data=attribute_data
    )

    # ================= extract and save region data ==================

    train_region = extract_region_data(train_data, region_data)
    val_region = extract_region_data(val_data, region_data)

    json.dump(train_region, open(os.path.join(args.output_dir, 'train_region.json'), 'w'))
    json.dump(val_region, open(os.path.join(args.output_dir, 'val_region.json'), 'w'))

    # ============== extract and save attribute ids ==================

    attribute_count = []
    for entry in train_data.values():
        for obj in entry['objects']:
            if 'attributes' in obj:
                attribute_count += obj['attributes']

    attribute_count = Counter(attribute_count).most_common(args.num_attributes)
    attribute2id = {j[0]: i for i, j in enumerate(attribute_count)}
    id2attribute = [j[0] for j in attribute_count]

    print_segment_line('saving attribute ids')
    json.dump(attribute2id, open(os.path.join(args.output_dir, 'attribute2id.json'), 'w'))
    json.dump(id2attribute, open(os.path.join(args.output_dir, 'id2attribute.json'), 'w'))

    # ============== extract and save relation ids ==================

    relation_count = []
    for entry in train_data.values():
        for rel in entry['relations']:
            relation_count.append(rel['predicate'])

    relation_count = Counter(relation_count).most_common(args.num_relations)
    relation2id = {j[0]: i for i, j in enumerate(relation_count)}
    id2relation = [j[0] for j in relation_count]

    print_segment_line('saving relation ids')
    json.dump(relation2id, open(os.path.join(args.output_dir, 'relation2id.json'), 'w'))
    json.dump(id2relation, open(os.path.join(args.output_dir, 'id2relation.json'), 'w'))

    split_dict = {'train': train_data, 'val': val_data}

    # ============== replace ids and save data ==================

    # compute ids for attributes and predicates
    for data in split_dict.values():
        for entry in data.values():
            for obj in entry['objects']:
                if 'attributes' in obj:
                    obj['attribute_ids'] = [attribute2id.get(x, len(attribute2id)) for x in obj['attributes']]

            for rel in entry['relations']:
                rel['predicate_id'] = relation2id.get(rel['predicate'], len(relation2id))

    print_segment_line('saving data')
    json.dump(train_data, open(os.path.join(args.output_dir, 'train.json'), 'w'))
    json.dump(val_data, open(os.path.join(args.output_dir, 'val.json'), 'w'))

    # ============== extract image features ==================

    print_segment_line('make directory for splits')

    for split in split_dict.keys():
        path = os.path.join(args.output_dir, split)
        if not os.path.isdir(path):
            os.mkdir(path)

    if args.image_dir is not None:
        for split, data in split_dict.items():
            data_list = list(data.values())
            print_segment_line('processing image data for {} set'.format(split))
            mp.spawn(
                main,
                args=(data_list, split, args),
                nprocs=args.gpu_num,
                join=True
            )
