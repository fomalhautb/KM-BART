from datetime import datetime

from torch.cuda.amp import autocast

import src.evaluation as vcg
from src.generation import generate_text
from src.utils import TaskType


def validate_pretraining_loss(
        epoch,
        model,
        val_loader,
        device,
        args,
        logger=None,
        log_interval=1,
        tb_writer=None
):
    total_step = len(val_loader)
    model.eval()
    loss = 0
    start_time = datetime.now()

    for i, batch in enumerate(val_loader):
        with autocast(enabled=args.amp):
            outputs = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch else None,
                decoder_attention_mask=batch['decoder_attention_mask'].to(
                    device) if 'decoder_attention_mask' in batch else None,
                labels=batch['labels'].to(device) if 'labels' in batch else None,
                mrm_labels=list(map(lambda x: x.to(device), batch['mrm_labels'])) if 'mrm_labels' in batch else None
            )

            loss += outputs[0]['loss'].item()

            if logger is not None and i % log_interval == 0:
                logger.info('Computing validation loss, Step [{}/{}], Loss: {:.4f}, ETA: {}'.format(
                    i + 1,
                    total_step,
                    loss / (i + 1),
                    str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
                ))

    loss /= total_step

    if logger is not None:
        logger.info('Validation loss', pad=True)
        logger.info('Epoch: {}, Val loss: {}'.format(
            epoch + 1,
            loss
        ))
        logger.line()

    if tb_writer is not None:
        tb_writer.add_scalars('loss/epoch', {'val': loss}, epoch + 1)


def validate_fine_tune_loss(
        epoch,
        model,
        val_loader,
        device,
        args,
        logger=None,
        log_interval=1,
        tb_writer=None
):
    total_step = len(val_loader)
    model.eval()
    loss = 0
    cls_loss = 0
    lm_loss = 0
    answer_loss = 0
    rationale_loss = 0
    start_time = datetime.now()

    for i, batch in enumerate(val_loader):
        with autocast(enabled=args.amp):
            outputs = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                decoder_input_ids=batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch else None,
                decoder_attention_mask=batch['decoder_attention_mask'].to(
                    device) if 'decoder_attention_mask' in batch else None,
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device) if 'labels' in batch else None,
                answer_ids=batch['answer_ids'].to(device) if 'answer_ids' in batch else None,
                answer_attention_mask=batch['answer_attention_mask'].to(
                    device) if 'answer_attention_mask' in batch else None,
            )

            loss += outputs[0].item()

            if logger is not None and i % log_interval == 0:
                logger.info('Computing validation loss, Step [{}/{}], Loss: {:.4f}, ETA: {}'.format(
                    i + 1,
                    total_step,
                    loss / (i + 1),
                    str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
                ))

    loss /= total_step
    cls_loss /= total_step
    lm_loss /= total_step
    answer_loss /= total_step
    rationale_loss /= total_step

    if logger is not None:
        logger.info('Validation loss', pad=True)
        logger.info('Epoch: {}, Val loss: {}'.format(
            epoch + 1,
            loss
        ))
        logger.line()

    if tb_writer is not None:
        tb_writer.add_scalars('loss/epoch', {'val': loss}, epoch + 1)


def validate_generation_score(
        epoch,
        model,
        gen_loader,
        reference,
        tokenizer,
        device,
        args,
        logger=None,
        log_interval=1,
        tb_writer=None,
):
    if not args.cpu:
        model = model.module

    generated = generate_text(
        model=model,
        gen_loader=gen_loader,
        tokenizer=tokenizer,
        device=device,
        args=args,
        logger=logger,
        log_interval=log_interval
    )

    scores = vcg.compute_metric_inference(gens_list=generated, refs_list=reference)

    if logger is not None:
        logger.info('Validation scores', pad=True)
        logger.info('Epoch: {}, BLEU2: {}, METEOR: {}, CIDEr: {}'.format(
            epoch + 1,
            scores['BLEU2'],
            scores['METEOR'],
            scores['CIDEr']
        ))
        logger.line()

    if tb_writer is not None:
        for k, v in scores.items():
            tb_writer.add_scalar('score/{}'.format(k), v, epoch + 1)
