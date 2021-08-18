import os
import sys

import torch

sys.path.append("comet-commonsense")
sys.path.remove(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive

sys.path.append(os.getcwd())


class AtomicGenerator:
    def __init__(self, args, rank=-1):
        self._opt, state_dict = interactive.load_model_file(args.model_file)
        os.chdir('comet-commonsense')
        self._data_loader, self._text_encoder = interactive.load_data("atomic", self._opt)
        os.chdir('..')

        n_ctx = self._data_loader.max_event + self._data_loader.max_effect
        n_vocab = len(self._text_encoder.encoder) + n_ctx
        self._model = interactive.make_model(self._opt, n_vocab, n_ctx, state_dict)
        self._sampling_algorithm = args.sampling_algorithm

        if rank != -1:
            cfg.device = rank
            cfg.do_gpu = True
            torch.cuda.set_device(cfg.device)
            self._model.cuda(cfg.device)
        else:
            cfg.device = "cpu"

        self._result_map = {"xIntent": "intent",
                            "xWant": "intent",
                            "xNeed": "before",
                            "xReact": "after",
                            "xEffect": "after"}

    def get_atomic_sequence(self, input_event, model, sampler, data_loader, text_encoder, category):
        if isinstance(category, list):
            outputs = {}
            for cat in category:
                new_outputs = self.get_atomic_sequence(
                    input_event, model, sampler, data_loader, text_encoder, cat)
                outputs.update(new_outputs)
            return outputs
        elif category == "all":
            outputs = {}

            for category in data_loader.categories:
                new_outputs = self.get_atomic_sequence(
                    input_event, model, sampler, data_loader, text_encoder, category)
                outputs.update(new_outputs)
            return outputs
        else:

            sequence_all = {}

            sequence_all["event"] = input_event
            sequence_all["effect_type"] = category

            with torch.no_grad():
                batch = interactive.set_atomic_inputs(
                    input_event, category, data_loader, text_encoder)

                sampling_result = sampler.generate_sequence(
                    batch, model, data_loader, data_loader.max_event +
                                               data.atomic_data.num_delimiter_tokens["category"],
                                               data_loader.max_effect -
                                               data.atomic_data.num_delimiter_tokens["category"])

            sequence_all['beams'] = sampling_result["beams"]

            # print_atomic_sequence(sequence_all)

            return {category: sequence_all}

    def get_reason(self, input_event):
        error_num = 0
        result = {"after": [],
                  "before": [],
                  "intent": []}
        category = ["xIntent", "xWant", "xNeed", "xReact", "xEffect"]
        sampler = interactive.set_sampler(self._opt, self._sampling_algorithm, self._data_loader)

        try:
            outputs = self.get_atomic_sequence(input_event, self._model, sampler, self._data_loader, self._text_encoder,
                                               category)
        except RuntimeError:
            outputs = {}
            error_num += 1

        for k in outputs.keys():
            if outputs[k]['beams'][0] != "none":
                result[self._result_map[k]].append(outputs[k]['beams'][0])
        return result
