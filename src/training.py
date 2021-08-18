from datetime import datetime

import numpy as np
from torch.cuda.amp import autocast

from src.utils import TaskType


def pretrain(
        epoch,
        model,
        train_loader,
        optimizer,
        device,
        args,
        logger=None,
        callback=None,
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
        # Forward pass
        with autocast(enabled=args.amp):
            outputs = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch else None,
                decoder_attention_mask=batch['decoder_attention_mask'].to(
                    device) if 'decoder_attention_mask' in batch else None,
                labels=batch['labels'].to(device) if 'labels' in batch else None,
                mrm_labels=list(map(lambda x: x.to(device), batch['mrm_labels'])) if 'mrm_labels' in batch else None,
                mrm_mask=batch['mrm_mask'].to(device) if 'mrm_mask' in batch else None,
                attribute_labels=list(
                    map(lambda x: x.to(device), batch['attribute_labels'])) if 'attribute_labels' in batch else None,
                attribute_mask=batch['attribute_mask'].to(device) if 'attribute_mask' in batch else None,
                relation_labels=batch['relation_labels'] if 'relation_labels' in batch else None
            )

            loss = outputs[0]['loss']

        total_loss += loss.item()

        # Backward and optimize
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
            tb_writer.add_scalars('loss/step', {'total loss': loss.item()}, step)

            for loss_name in outputs[0]:
                if loss_name != 'loss':
                    tb_writer.add_scalars('loss/step', {loss_name.replace('_', ' '): outputs[0][loss_name].item()},
                                          step)

        if callback is not None:
            callback(
                step=i,
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                args=args,
                logger=logger
            )

    if tb_writer is not None:
        tb_writer.add_scalars('loss/epoch', {'train': total_loss / total_step}, epoch + 1)


def fine_tune(
        epoch,
        model,
        train_loader,
        optimizer,
        device,
        args,
        logger=None,
        callback=None,
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
        # Forward pass
        with autocast(enabled=args.amp):
            outputs = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch else None,
                decoder_attention_mask=batch['decoder_attention_mask'].to(
                    device) if 'decoder_attention_mask' in batch else None,
                labels=batch['labels'].to(device),
                answer_ids=batch['answer_ids'].to(device) if 'answer_ids' in batch else None,
                answer_attention_mask=batch['answer_attention_mask'].to(
                    device) if 'answer_attention_mask' in batch else None,
            )

            loss = outputs[0]

        total_loss += loss.item()
        # Backward and optimize
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
            tb_writer.add_scalars('loss/step', {'loss': loss.item()}, step)

        if callback is not None:
            callback(
                step=i,
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                args=args,
                logger=logger
            )

    if tb_writer is not None:
        tb_writer.add_scalars('loss/epoch', {'train': total_loss / total_step}, epoch + 1)
