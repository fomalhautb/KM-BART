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

# from .feature_extractor import FeatureExtractor
from .atomic_generator import AtomicGenerator
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
    new_cap = new_cap.split('@')[0]  # remove everything after @
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
    except (IOError, SyntaxError) as e:
        os.remove(image_dir)
        print('Deleted corrupt image:', image_dir)


def download_image(index, url, path):
    image_dir = os.path.join(path, str(index) + '.jpg')

    if os.path.isfile(image_dir):
        return

    try:
        response = requests.get(url, stream=False, timeout=5, allow_redirects=True, headers=headers)
        with open(image_dir, 'wb') as file:
            response.raw.decode_content = True
            file.write(response.content)
    except:
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
            'caption': clean_caption(caption)
        }

    return None


def get_text_data(entry, index, atomic_generator):
    data = []
    base_entry = {
        'img_id': str(entry['img_id']),
        'img_fn': entry['img_fn'],
        'index': index,
    }

    res = atomic_generator.get_reason(entry['caption'])
    # if len(res['before']) == 0 or len(res['after']) == 0 or len(res['intent']) == 0:
    #     return []
    for k in ['before', 'after', 'intent']:
        for ans in res[k]:
            data.append({**base_entry, 'event': entry['caption'], 'task_type': k, 'labels': ans})

    return data, res


def get_eval_data(entry, index, ref_ans):
    data = []
    base_entry = {
        'img_id': str(entry['img_id']),
        'img_fn': entry['img_fn'],
        'index': index,
    }
    if len(ref_ans['before']) != 0:
        data.append({**base_entry, 'event': entry['caption'], 'task_type': 'before'})
    if len(ref_ans['after']) != 0:
        data.append({**base_entry, 'event': entry['caption'], 'task_type': 'after'})
    if len(ref_ans['intent']) != 0:
        data.append({**base_entry, 'event': entry['caption'], 'task_type': 'intent'})
    return data


def get_reference_data(entry, ref_ans):
    data = []
    # if len(ref_ans['before']) == 0 or len(ref_ans['after']) == 0 or len(ref_ans['intent']) == 0:
    #     return []
    data.append(ref_ans)
    return data


def process_text(rank, args, split, split_dict):
    local_data = split_dict[rank::args.gpu_num]
    text_load_path = os.path.join(args.output_dir, split + str(rank) + '.json')
    eval_load_path = os.path.join(args.output_dir, split + str(rank) + '_eval.json')
    ref_load_path = os.path.join(args.output_dir, split + str(rank) + '_ref.json')
    if os.path.exists(text_load_path) and os.path.exists(eval_load_path) and os.path.exists(ref_load_path):
        text_data = json.load(open(text_load_path, 'r'))
        eval_data = json.load(open(eval_load_path, 'r'))
        ref_data = json.load(open(ref_load_path, 'r'))
        start_idx = text_data[-1]['index']
    else:
        text_data = []
        eval_data = []
        ref_data = []
        start_idx = -1
    start_time = datetime.now()
    generator = AtomicGenerator(args, rank)
    print(start_idx)

    for i in range(start_idx + 1, len(local_data), 1):
        entry = local_data[i]
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
        if i % 10000 == 0:
            json.dump(text_data, open(os.path.join(args.output_dir, split + str(rank) + '.json'), 'w'))
            json.dump(eval_data, open(os.path.join(args.output_dir, split + str(rank) + '_eval.json'), 'w'))
            json.dump(ref_data, open(os.path.join(args.output_dir, split + str(rank) + '_ref.json'), 'w'))

    json.dump(text_data, open(os.path.join(args.output_dir, split + str(rank) + '.json'), 'w'))
    json.dump(eval_data, open(os.path.join(args.output_dir, split + str(rank) + '_eval.json'), 'w'))
    json.dump(ref_data, open(os.path.join(args.output_dir, split + str(rank) + '_ref.json'), 'w'))


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

    for i, entry in enumerate(local_data):
        image_data = get_image_data(entry=entry, data_dir=args.data_dir, extractor=extractor)
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
    parser.add_argument('--download', action='store_true',
                        help='Download images')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='download/load images from this directory')
    parser.add_argument('--no_img_feat', action='store_true',
                        help='not generate image features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--annot_dir', type=str, required=True,
                        help='SBU annotation directory with '
                             '"SBU_captioned_photo_dataset_captions.txt", and "SBU_captioned_photo_dataset_urls.txt"')
    parser.add_argument('--max_index', type=int, default=-1,
                        help='The maximum index')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='number of jobs for downloading')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='proportion of training set')
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='number of GPUs in total')
    parser.add_argument('--config', type=str, default='config/extract_config.yaml',
                        help='path extractor config')
    parser.add_argument("--model_file", type=str,
                        default="comet-commonsense/model/pretrained_models/atomic_pretrained_model.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="beam-1")

    args = parser.parse_args()

    if args.download and args.data_dir is None:
        raise ValueError('if --download is set, --data_dir must be specified')

    with open(os.path.join(args.annot_dir, 'SBU_captioned_photo_dataset_captions.txt')) as f:
        captions = f.readlines()

    with open(os.path.join(args.annot_dir, 'SBU_captioned_photo_dataset_urls.txt')) as f:
        urls = f.readlines()

    if args.max_index == -1:
        args.max_index = max(len(captions), len(urls))

    start = datetime.now()
    if args.download:
        # download images
        Parallel(n_jobs=args.n_jobs)(
            delayed(download_image)(index, url, args.data_dir)
            for index, url in enumerate(tqdm(urls[:args.max_index]))
        )

        # delete invalid images
        Parallel(n_jobs=args.n_jobs)(
            delayed(delete_invalid)(index, args.data_dir)
            for index in tqdm(range(len(urls[:args.max_index])))
        )
        print_segment_line("Download complete in: " + str(datetime.now() - start))

    start = datetime.now()

    # build index of valid images
    raw_data = Parallel(n_jobs=args.n_jobs)(
        delayed(build_index)(index, caption, args.data_dir)
        for index, caption in enumerate(tqdm(captions[:args.max_index]))
    )

    raw_data = list(filter(lambda x: x is not None, raw_data))

    split_index = int(len(raw_data) * args.train_ratio)
    train_data = raw_data[:split_index]
    val_data = raw_data[split_index:]

    # json.dump(train_data, open(os.path.join(args.output_dir, 'train.json'), 'w'))
    # json.dump(val_data, open(os.path.join(args.output_dir, 'val.json'), 'w'))

    split_dict = {'train': train_data, 'val': val_data}

    print_segment_line("Build index complete in: " + str(datetime.now() - start))

    for split, data in split_dict.items():
        print_segment_line('generate comet reason for {} set'.format(split))
        mp.spawn(
            process_text,
            args=(args, split, data),
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

    # make directory for splits
    for split in split_dict.keys():
        path = os.path.join(args.output_dir, split)
        if not os.path.isdir(path):
            os.mkdir(path)

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
