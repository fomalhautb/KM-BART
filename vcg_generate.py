import argparse
import json
from datetime import datetime

from torch.utils.data import DataLoader

from src.data.collation import Collator
from src.data.dataset import VCGDataset
from src.data.tokenization import ConditionTokenizer
from src.generation import generate_text
from src.model.model import MultiModalBartForConditionalGeneration
from src.utils import Logger


def main(args):
    device = 'cpu' if args.cpu else 'cuda'
    logger = Logger(log_dir=args.log_dir, enabled=True)

    logger.info('Loading model...')

    tokenizer = ConditionTokenizer()
    collate_fn = Collator(tokenizer, has_label=False)

    # load checkpoint
    model = MultiModalBartForConditionalGeneration.from_pretrained(args.checkpoint)
    model.to(device)
    logger.info('Loaded model from "{}"'.format(args.checkpoint))

    logger.info('Loading data...')

    dataset = VCGDataset(
        args.data_dir,
        split=args.split,
        use_image=args.use_image,
        use_event=args.use_event,
        eval_mode=True
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    start = datetime.now()

    logger.info('Start generation', pad=True)

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        gen_loader=loader,
        args=args,
        device=device,
        logger=logger,
        log_interval=1
    )

    logger.info("Generation complete in: " + str(datetime.now() - start), pad=True)
    logger.info('Saving results...')

    with open(args.output_file, 'w') as outfile:
        json.dump(generated, outfile)

    logger.info('Saved results in "{}"'.format(args.output_file))


def parse_args():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('--data_dir', required=True, type=str,
                        help='path to load data, output_dir of prepare_vcg')
    parser.add_argument('--output_file', required=True, type=str,
                        help='file to save the generated result')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='load model from checkpoint')

    # path
    parser.add_argument('--log_dir', default=None, type=str,
                        help='path to output log files, do not output to file if not specified')
    parser.add_argument('--split', default='val', type=str,
                        help='generate for which split')

    # model
    parser.add_argument('--no_event', dest='use_event', action='store_false',
                        help='not to use event descriptions')
    parser.add_argument('--no_image', dest='use_image', action='store_false',
                        help='not to use image features')
    parser.add_argument('--model', type=str, default='base',
                        help='base or large bart')

    # evaluation
    parser.add_argument('--num_gen', default=1, type=int,
                        help='number of generated sentence')
    parser.add_argument('--num_beams', default=1, type=int,
                        help='level of beam search')
    parser.add_argument('--do_sample', action='store_true',
                        help='use nucleus sample')
    parser.add_argument('--top_p', default=1.0, type=float,
                        help='top p for generation')
    parser.add_argument('--top_k', default=0, type=int,
                        help='top k for generation')

    # hardware and performance
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='number of GPUs in total')
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
