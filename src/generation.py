from datetime import datetime

from torch.cuda.amp import autocast


def generate_text(
        model,
        gen_loader,
        tokenizer,
        args,
        device,
        logger=None,
        log_interval=1,
):
    total_step = len(gen_loader)
    model.eval()
    generated = []
    start_time = datetime.now()

    for i, batch in enumerate(gen_loader):
        with autocast(enabled=args.amp):
            outputs = model.generate(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                num_beams=args.num_beams,
                num_return_sequences=args.num_gen,
                do_sample=args.do_sample if hasattr(args, 'do_sample') else False,
                top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
                top_k=args.top_k if hasattr(args, 'top_k') else 0,
                early_stopping=True
            )
        # decode generated sentences and append to "generated"
        for j in range(len(batch['index'])):
            generations = []
            for output in outputs[j * args.num_gen: (j + 1) * args.num_gen]:
                generations.append(tokenizer.decode(output, skip_special_tokens=True))

            generated.append({
                'index': batch['index'][j],
                'task_type': batch['task_type'][j],
                'generations': generations
            })

        if (i + 1) % log_interval == 0:
            logger.info('Generating, Step [{}/{}], ETA: {}'.format(
                i + 1,
                total_step,
                str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
            ))

    return generated
