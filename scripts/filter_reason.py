import argparse
import json
import os
from datetime import datetime

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from src.data.collation import Collator
from src.data.dataset import ReasonDataset
from src.data.tokenization import ConditionTokenizer
from src.model.model import MultiModalBartForConditionalGeneration
from src.utils import Logger


def perplexity(pred, label):
    lsoftmax = torch.nn.LogSoftmax(dim=1)
    softmax_prob = lsoftmax(pred)
    prob = torch.Tensor([softmax_prob[i][j] for i, j in enumerate(label) if j >= 0])
    return torch.exp(-prob.mean())


def filter(model, loader, device, args, logger):
    filtered_indices = []
    total_step = len(loader)
    model.eval()
    start_time = datetime.now()

    for i, batch in enumerate(loader):
        with autocast(enabled=args.amp):
            outputs = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device),
                decoder_attention_mask=batch['decoder_attention_mask'].to(device),
                use_cache=False
            )

        for j in range(len(batch['input_ids'])):
            pp = perplexity(outputs[0][j], batch['labels'][j])
            if torch.log(pp) < args.pp_threshold:
                filtered_indices.append(batch['dataset_index'][j])

        logger.info('Filtering, Step [{}/{}], ETA: {}'.format(
            i + 1,
            total_step,
            str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
        ))

    return filtered_indices


def main(args):
    device = 'cpu' if args.cpu else 'cuda'
    logger = Logger(log_dir=args.log_dir, enabled=True)

    logger.info('Loading model...')

    tokenizer = ConditionTokenizer()
    collate_fn = Collator(tokenizer, has_label=True)

    # load checkpoint
    model = MultiModalBartForConditionalGeneration.from_pretrained(args.checkpoint)
    model.to(device)
    logger.info('Loaded model from "{}"'.format(args.checkpoint))

    logger.info('Loading data...')

    train_dataset = ReasonDataset(
        args.data_dir,
        split=args.split
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    start = datetime.now()

    logger.info('Start computing score', pad=True)

    train_filtered_indices = filter(model, train_loader, device, args, logger)

    logger.info("Filtering complete in: " + str(datetime.now() - start), pad=True)
    logger.info('Saving results...')

    train_data = [train_dataset.get_raw_data(i) for i in train_filtered_indices]
    json.dump(train_data, open(os.path.join(args.output_dir, f'reason_{args.split}.json'), 'w'))

    logger.info(f'Remaining {len(train_data)}/{len(train_dataset)}')

    logger.info('Saved results in "{}"'.format(args.output_dir))


def parse_args():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('--data_dir', required=True, type=str,
                        help='path to load data')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='dir to save the filtered result')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='load model from checkpoint')

    # path
    parser.add_argument('--log_dir', default=None, type=str,
                        help='path to output log files, do not output to file if not specified')
    parser.add_argument('--split', default='train', type=str,
                        help='generate for which split')

    # filtering
    parser.add_argument('--pp_threshold', default=3.5, type=float,
                        help='perplexity threshold for filtering')

    # hardware and performance
    parser.add_argument('--cpu', action='store_true',
                        help='if only use cpu to run the model')
    parser.add_argument('--amp', action='store_true',
                        help='whether or not to use amp')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='#workers for data loader')

    parser.set_defaults(use_event=True, use_image=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
