import argparse
import json
import os
import pickle
import re
import warnings
from datetime import datetime

import cv2
import requests
import torch.multiprocessing as mp
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm

from .feature_extractor import FeatureExtractor
from .utils import print_segment_line

headers = {
    'User-Agent': 'Googlebot-Image/1.0',
    'X-Forwarded-For': '64.18.15.200'
}


def clean_caption(cap):
    new_cap = cap
    new_cap = new_cap.replace(r'&amp;', ' ').replace(r'quot;', ' ').replace('amp;', ' ')  # remove html tags
    new_cap = re.sub(r'\([^>]+?\)', '', new_cap)  # remove everything in (...)
    new_cap = re.sub(r'\.+', '.', new_cap)  # remove redundant dots
    new_cap = re.sub(r'[^\S\n\t]+', ' ', new_cap)  # remove redundant spacing
    new_cap = new_cap.strip()
    return new_cap


def delete_invalid(index, path):
    image_dir = os.path.join(path, str(index) + '.jpg')

    if not os.path.isfile(image_dir):
        return

    try:
        img = Image.open(image_dir)
        img.verify()
        assert img.size[0] > 50 and img.size[1] > 50
    except (IOError, ValueError, AssertionError) as e:
        os.remove(image_dir)
        print('Deleted corrupt image:', image_dir, flush=True)


def download_image(index, url, path):
    image_dir = os.path.join(path, str(index) + '.jpg')

    if os.path.isfile(image_dir):
        return

    try:
        response = requests.get(url, stream=False, timeout=5, allow_redirects=True, headers=headers)
        with open(image_dir, 'wb') as file:
            response.raw.decode_content = True
            file.write(response.content)
    except Exception as e:
        print('failed to download {}'.format(url), flush=True)


def build_index(index, caption, data_dir):
    image_file = os.path.join(data_dir, str(index) + '.jpg')
    img = cv2.imread(image_file)

    if img is not None:  # check if image is valid
        return {
            'img_id': index,
            'img_fn': str(index) + '.jpg',
            'width': img.shape[1],
            'height': img.shape[0],
            'labels': clean_caption(caption)
        }

    return None


def get_image_data(entry, data_dir, extractor):
    img_dir = entry['img_fn']
    im = cv2.imread(os.path.join(data_dir, img_dir))
    features = extractor.extract_feature(im)

    return {
        'image_features': features['features'],
        'mrm_labels': features['scores'],
        'boxes': features['boxes']
    }


def main(rank, data, split, args):
    extractor = FeatureExtractor(args.config, rank)
    local_data = data[rank::args.gpu_num]
    start_time = datetime.now()
    data_dir = os.path.join(args.data_dir, split)

    for i, entry in enumerate(local_data):
        save_path = os.path.join(args.output_dir, split, str(entry['img_id']) + '.pkl')
        if os.path.isfile(save_path) and args.skip_generated:
            print('Skipped {}'.format(entry['img_fn']), flush=True)
            continue
        image_data = get_image_data(entry=entry, data_dir=data_dir, extractor=extractor)
        pickle.dump(image_data, open(save_path, 'wb'))
        print('GPU{}, {}/{}, ETA: {}'.format(
            rank,
            i,
            len(local_data),
            str((len(local_data) - (i + 1)) / (i + 1) * (datetime.now() - start_time))
        ), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true',
                        help='Download images')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='download/load images from this directory')
    parser.add_argument('--no_img_feat', action='store_true',
                        help='not generate image features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--annot_dir', type=str, required=True,
                        help='CC annotation directory with '
                             '"Train_GCC-training.tsv", and "Validation_GCC-1.1.0-Validation.tsv"')
    parser.add_argument('--max_index', type=int, default=-1,
                        help='The maximum index')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='number of jobs for downloading')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='proportion of training set')
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='number of GPUs in total')
    parser.add_argument('--config', type=str, default=None,
                        help='path extractor config')
    parser.add_argument('--delete_invalid', action='store_true',
                        help='Delete invalid images in data_dir')
    parser.add_argument('--skip_generated', action='store_true',
                        help='skip generated image features')
    args = parser.parse_args()

    if args.download and args.data_dir is None:
        raise ValueError('if --download is set, --data_dir must be specified')

    if args.config is None:
        args.config = 'config/extract_config.yaml'

    with open(os.path.join(args.annot_dir, 'Train_GCC-training.tsv')) as f:
        train_file = [list(map(lambda x: x.strip(), line.split('\t'))) for line in f.readlines()]

    with open(os.path.join(args.annot_dir, 'Validation_GCC-1.1.0-Validation.tsv')) as f:
        val_file = [list(map(lambda x: x.strip(), line.split('\t'))) for line in f.readlines()]

    split_dict = {'train': train_file, 'val': val_file}

    # make directory for splits
    for split in split_dict.keys():
        path = os.path.join(args.data_dir, split)
        if not os.path.isdir(path):
            os.mkdir(path)

    start = datetime.now()
    for split, data_file in split_dict.items():
        path = os.path.join(args.data_dir, split)
        urls = [x[1] for x in data_file]

        if args.download:
            # download images
            Parallel(n_jobs=args.n_jobs)(
                delayed(download_image)(index, url, path)
                for index, url in enumerate(tqdm(urls[:args.max_index]))
            )

        if args.delete_invalid:
            # delete invalid images
            Parallel(n_jobs=args.n_jobs)(
                delayed(delete_invalid)(index, path)
                for index in tqdm(range(len(urls[:args.max_index])))
            )

    # build index of valid images
    train_captions = [x[0] for x in train_file]
    train_data = Parallel(n_jobs=args.n_jobs)(
        delayed(build_index)(index, caption, os.path.join(args.data_dir, 'train'))
        for index, caption in enumerate(tqdm(train_captions[:args.max_index]))
    )
    train_data = list(filter(lambda x: x is not None, train_data))

    val_captions = [x[0] for x in val_file]
    val_data = Parallel(n_jobs=args.n_jobs)(
        delayed(build_index)(index, caption, os.path.join(args.data_dir, 'val'))
        for index, caption in enumerate(tqdm(val_captions[:args.max_index]))
    )
    val_data = list(filter(lambda x: x is not None, val_data))

    json.dump(train_data, open(os.path.join(args.output_dir, 'train.json'), 'w'))
    json.dump(val_data, open(os.path.join(args.output_dir, 'val.json'), 'w'))

    split_dict = {'train': train_data, 'val': val_data}

    # make directory for splits
    for split in split_dict.keys():
        path = os.path.join(args.output_dir, split)
        if not os.path.isdir(path):
            os.mkdir(path)

    print_segment_line("Build index complete in: " + str(datetime.now() - start))

    start = datetime.now()
    if not args.no_img_feat:
        for split, data in split_dict.items():
            print_segment_line('extracting image features for {} set'.format(split))
            mp.spawn(
                main,
                args=(data, split, args),
                nprocs=args.gpu_num,
                join=True
            )
        print_segment_line("Build features complets in " + str(datetime.now() - start))
