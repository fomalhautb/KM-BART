import argparse
import json
import os
from datetime import datetime

import torch
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

from src.data.collation import Collator
from src.data.dataset import VCGDataset
from src.data.tokenization import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.model import MultiModalBartForConditionalGeneration
from src.training import fine_tune
from src.utils import (
    Logger,
    save_training_data,
    load_training_data,
    setup_process,
    cleanup_process
)
from src.validation import validate_fine_tune_loss, validate_generation_score


def main(rank, args):
    # ============ logging, initialization and directories ==============

    if not args.cpu:
        setup_process(rank, args.gpu_num, master_port=args.master_port)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_path = os.path.join(args.checkpoint_dir, timestamp)
    tb_writer = None
    log_dir = os.path.join(args.log_dir, timestamp)

    # make log dir and tensorboard writer if log_dir is specified
    if rank == 0 and args.log_dir is not None:
        os.makedirs(log_dir)
        tb_writer = SummaryWriter(log_dir=log_dir)

    logger = Logger(log_dir=os.path.join(log_dir, 'log.txt'), enabled=(rank == 0))

    # make checkpoint dir if not exist
    if rank == 0 and not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
        logger.info('Made checkpoint directory: "{}"'.format(checkpoint_path))

    logger.info('Initialed with {} GPU(s)'.format(args.gpu_num), pad=True)
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))

    # =========================== model =============================

    logger.info('Loading model...')

    if args.cpu:
        device = 'cpu'
        map_location = device
    else:
        device = torch.device("cuda:{}".format(rank))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    tokenizer = ConditionTokenizer()

    if args.model_config is not None:
        bart_config = MultiModalBartConfig.from_dict(json.load(open(args.model_config)))
    else:
        bart_config = MultiModalBartConfig.from_pretrained(args.checkpoint)

    if args.dropout is not None:
        bart_config.dropout = args.dropout
    if args.attention_dropout is not None:
        bart_config.attention_dropout = args.attention_dropout
    if args.classif_dropout is not None:
        bart_config.classif_dropout = args.classif_dropout
    if args.activation_dropout is not None:
        bart_config.activation_dropout = args.activation_dropout

    if args.checkpoint:
        model = MultiModalBartForConditionalGeneration.from_pretrained(
            args.checkpoint,
            config=bart_config,
            error_on_mismatch=False
        )
    else:
        model = MultiModalBartForConditionalGeneration(bart_config)

    model.to(device)

    if not args.cpu:
        torch.cuda.set_device(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler() if args.amp else None

    epoch = 0
    if args.continue_training:
        epoch = load_training_data(
            args.checkpoint,
            optimizer=optimizer,
            scaler=scaler,
            map_location=map_location
        )['epoch'] + 1

    # =========================== data =============================

    logger.info('Loading data...')

    collate_fn = Collator(tokenizer, has_label=True)
    collate_fn_gen = Collator(tokenizer, has_label=False)

    # training set
    train_dataset = VCGDataset(
        args.data_dir,
        split='train',
        use_image=args.use_image,
        use_event=args.use_event
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.gpu_num,
        rank=rank
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn
    )

    # validation set
    val_dataset = VCGDataset(
        args.data_dir,
        split='val',
        use_image=args.use_image,
        use_event=args.use_event
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # validation set for generation
    gen_dataset = VCGDataset(
        args.data_dir,
        split='val',
        use_image=args.use_image,
        use_event=args.use_event,
        eval_mode=True
    )

    gen_loader = DataLoader(
        dataset=gen_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_gen
    )

    val_ref = json.load(open(os.path.join(args.data_dir, 'val_ref.json'), 'r'))

    # ========================== training ============================

    # generate test examples
    def callback(step, **kwargs):
        if logger is not None and (step + 1) % 100 == 0:
            data = train_dataset[0]
            inputs = collate_fn([data])
            input_ids = inputs['input_ids'].to(device)
            image_features = list(map(lambda x: x.to(device), inputs['image_features']))
            generation_model = model if args.cpu else model.module
            generated = generation_model.generate(input_ids, max_length=100, image_features=image_features)
            ans = tokenizer.decode(generated[0], skip_special_tokens=True)
            event = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            logger.info('Input ({} image): "{}"'.format('with' if args.use_image else 'without', event))
            logger.info('Generated: "{}"'.format(ans))

    logger.info('Start training', pad=True)

    start = datetime.now()

    while epoch < args.epochs:
        logger.info('Epoch {}'.format(epoch + 1), pad=True)

        fine_tune(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            args=args,
            device=device,
            logger=logger,
            callback=callback,
            log_interval=1,
            tb_writer=tb_writer,
            tb_interval=1,
            scaler=scaler
        )

        # save checkpoint
        if rank == 0:
            logger.info('Validating Epoch {}'.format(epoch + 1), pad=True)
            # save memory and faster with no_grad()
            with torch.no_grad():
                if args.validate_loss:
                    validate_fine_tune_loss(
                        epoch=epoch,
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        args=args,
                        logger=logger,
                        log_interval=1,
                        tb_writer=tb_writer
                    )

                if args.validate_score:
                    validate_generation_score(
                        epoch=epoch,
                        model=model,
                        gen_loader=gen_loader,
                        reference=val_ref,
                        tokenizer=tokenizer,
                        device=device,
                        args=args,
                        logger=logger,
                        log_interval=1,
                        tb_writer=tb_writer,
                    )

            current_checkpoint_path = os.path.join(checkpoint_path, 'model{}'.format(epoch))

            if args.cpu:
                model.save_pretrained(current_checkpoint_path)
            else:
                model.module.save_pretrained(current_checkpoint_path)

            save_training_data(
                path=current_checkpoint_path,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch
            )
            logger.info('Saved checkpoint at "{}"'.format(checkpoint_path))

        epoch += 1

    logger.info("Training complete in: " + str(datetime.now() - start), pad=True)

    if not args.cpu:
        cleanup_process()


def parse_args():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('--data_dir', required=True, type=str,
                        help='path to load data, output_dir of prepare_vcg')
    parser.add_argument('--checkpoint_dir', required=True, type=str,
                        help='where to save the checkpoint')

    # path
    parser.add_argument('--log_dir', default=None, type=str,
                        help='path to output log files, not output to file if not specified')
    parser.add_argument('--model_config', default=None, type=str,
                        help='path to load model config')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='name or path to load weights')

    # model
    parser.add_argument('--no_event', dest='use_event', action='store_false',
                        help='not to use event descriptions')
    parser.add_argument('--no_image', dest='use_image', action='store_false',
                        help='not to use image features')

    # training and evaluation
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of training epoch')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--num_gen', default=1, type=int,
                        help='number of generated sentence on validation.')
    parser.add_argument('--num_beams', default=1, type=int,
                        help='level of beam search on validation')
    parser.add_argument('--continue_training', action='store_true',
                        help='continue training, load optimizer and epoch from checkpoint')
    parser.add_argument('--validate_loss', action='store_true',
                        help='compute the validation loss at the end of each epoch')
    parser.add_argument('--validate_score', action='store_true',
                        help='compute the validation score (BLEU, METEOR, etc.) at the end of each epoch')

    # dropout
    parser.add_argument('--dropout', default=None, type=float,
                        help='dropout rate for the transformer. This overwrites the model config')
    parser.add_argument('--classif_dropout', default=None, type=float,
                        help='dropout rate for the classification layers. This overwrites the model config')
    parser.add_argument('--attention_dropout', default=None, type=float,
                        help='dropout rate for the attention layers. This overwrites the model config')
    parser.add_argument('--activation_dropout', default=None, type=float,
                        help='dropout rate for the activation layers. This overwrites the model config')

    # hardware and performance
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='number of GPUs in total')
    parser.add_argument('--cpu', action='store_true',
                        help='if only use cpu to run the model')
    parser.add_argument('--amp', action='store_true',
                        help='whether or not to use amp')
    parser.add_argument('--master_port', type=str, default='12355',
                        help='master port for DDP')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='#workers for data loader')

    parser.set_defaults(use_event=True, use_image=True)
    args = parser.parse_args()

    if args.gpu_num != 1 and args.cpu:
        raise ValueError('--gpu_num are not allowed if --cpu is set to true')

    if args.checkpoint is None and args.model_config is None:
        raise ValueError('--model_config and --checkpoint cannot be empty at the same time')

    return args


if __name__ == '__main__':
    args = parse_args()

    mp.spawn(
        main,
        args=(args,),
        nprocs=args.gpu_num,
        join=True
    )
