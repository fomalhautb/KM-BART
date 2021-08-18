import argparse
import os
from datetime import datetime

import torch
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from transformers import BertTokenizer, BertModel

from src.data.collation import AtomicCollator
from src.data.dataset import VCGDataset
from src.model.model import ReasoningClassification
from src.utils import (
    Logger,
    save_training_data,
    load_training_data,
    setup_process,
    cleanup_process
)


def fine_tune(
        epoch,
        model,
        train_loader,
        optimizer,
        device,
        args,
        logger=None,
        log_interval=1,
        tb_writer=None,
        tb_interval=1,
        scaler=None
):
    total_step = len(train_loader)
    model.train()
    total_loss = 0

    start_time = datetime.now()

    for i, batch in enumerate(train_loader):
        # print(batch['text'].size())
        # print(batch['image'].size())
        # print(batch['label'].size())
        # print(batch['image'])
        with autocast(enabled=args.amp):
            loss = model.forward(
                txt=batch['text'].to(device),
                image=batch['image'].to(device),
                label=batch['label'].to(device),
            )

        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if logger is not None and i % log_interval == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, ETA: {}'.format(
                epoch + 1,
                args.epochs,
                i + 1,
                total_step,
                loss.item(),
                str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
            ))

        if tb_writer is not None and i % tb_interval == 0:
            step = epoch * total_step + i + 1
            tb_writer.add_scalars('Loss/Step', {'Total loss': loss.item()}, step)

    if tb_writer is not None:
        tb_writer.add_scalars('Loss/Epoch', {'Train': total_loss / total_step}, epoch + 1)


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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    backbone = BertModel.from_pretrained('bert-base-uncased')
    txt_seq_len = 30
    image_seq_len = 30
    model = ReasoningClassification(txt_dim=txt_seq_len * 768, image_dim=image_seq_len * 2052, inner_dim=1024)

    if args.checkpoint:
        model.load_state_dict(torch.load(checkpoint_path))

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

    collate_fn = AtomicCollator(
        tokenizer=tokenizer,
        txt_backbone=backbone,
        txt_seq_length=txt_seq_len,
        image_seq_length=image_seq_len,
        shuffle_ratio=0.5
    )

    # training set
    train_dataset = VCGDataset(
        args.data_dir,
        split='train'
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

    # ========================== training ============================

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
            log_interval=1,
            tb_writer=tb_writer,
            tb_interval=1,
            scaler=scaler
        )

        epoch += 1

    current_checkpoint_path = os.path.join(checkpoint_path, 'model{}'.format(epoch))

    save_training_data(
        path=current_checkpoint_path,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch
    )

    torch.save(model.state_dict(), checkpoint_path, 'model_state.pt')

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
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='name or path to load weights')

    # training and evaluation
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of training epoch')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--num_gen', default=1, type=int,
                        help='number of generated sentence on validation')
    parser.add_argument('--continue_training', action='store_true',
                        help='continue training, load optimizer and epoch from checkpoint')

    # dropout
    parser.add_argument('--dropout', default=None, type=float,
                        help='dropout rate for the transformer. This overwrites the model config')

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

    parser.set_defaults(use_event=True, use_image=True, use_boxes=True)
    args = parser.parse_args()

    if args.gpu_num != 1 and args.cpu:
        raise ValueError('--gpu_num are not allowed if --cpu is set to true')

    return args


if __name__ == '__main__':
    args = parse_args()

    mp.spawn(
        main,
        args=(args,),
        nprocs=args.gpu_num,
        join=True
    )
