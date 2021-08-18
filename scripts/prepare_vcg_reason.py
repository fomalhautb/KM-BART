import argparse
import json
import os
import warnings
from datetime import datetime

import torch.multiprocessing as mp

# from .feature_extractor import FeatureExtractor
from .atomic_generator import AtomicGenerator
from .utils import print_segment_line


def get_text_data(entry, index, atomic_generator):
    data = []
    base_entry = {
        'img_id': str(entry['img_id']),
        'img_fn': entry['img_fn'],
        'index': index,
    }

    res = atomic_generator.get_reason(entry['event'])
    for k in ['before', 'after', 'intent']:
        for ans in res[k]:
            data.append({**base_entry, 'event': entry['event'], 'task_type': k, 'labels': ans})

    return data, res


def get_eval_data(entry, index, ref_ans):
    data = []
    base_entry = {
        'img_id': str(entry['img_id']),
        'img_fn': entry['img_fn'],
        'index': index,
    }

    if len(ref_ans['before']) != 0:
        data.append({**base_entry, 'event': entry['event'], 'task_type': 'before'})
    if len(ref_ans['after']) != 0:
        data.append({**base_entry, 'event': entry['event'], 'task_type': 'after'})
    if len(ref_ans['intent']) != 0:
        data.append({**base_entry, 'event': entry['event'], 'task_type': 'intent'})
    return data


def get_reference_data(entry, ref_ans):
    data = []
    data.append(ref_ans)
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


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--annot_dir', type=str, required=True,
                        help='VCG annotation directory with "captions_train2014.json", '
                             '"captions_val2014.json", "instances_train2014.json" and "instances_val2014.json"')
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='number of GPUs in total')
    parser.add_argument("--model_file", type=str,
                        default="comet-commonsense/model/pretrained_models/atomic_pretrained_model.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="beam-1")

    args = parser.parse_args()

    print_segment_line('extracting training annotations')

    raw_data = json.load(open(os.path.join(args.annot_dir, 'train.json'), 'r'))[:21]
    train_data = [raw_data[0]]
    for i in range(1, len(raw_data), 1):
        if (raw_data[i]['event'] != raw_data[i - 1]['event']) or (raw_data[i]['img_id'] != raw_data[i - 1]['img_id']):
            raw_data[i]['index'] = len(train_data)
            train_data.append(raw_data[i])

    print_segment_line('extracting validation annotations')

    raw_data = json.load(open(os.path.join(args.annot_dir, 'val.json'), 'r'))[:21]
    val_data = [raw_data[0]]
    for i in range(1, len(raw_data), 1):
        if (raw_data[i]['event'] != raw_data[i - 1]['event']) or (raw_data[i]['img_id'] != raw_data[i - 1]['img_id']):
            raw_data[i]['index'] = len(val_data)
            val_data.append(raw_data[i])

    split_dict = {'train': train_data, 'val': val_data}

    print_segment_line('generating textual and reference data')

    # generate and save training data, evaluation data and reference data
    for split, data in split_dict.items():
        print_segment_line('generate comet reason for {} set'.format(split))
        mp.spawn(
            process_text,
            args=(args, split, list(data)),
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
