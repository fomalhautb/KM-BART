import argparse
import json

from src.evaluation import compute_metric_inference
from src.utils import Logger


def main(args):
    logger = Logger()

    with open(args.generation, 'r') as json_file:
        gens_list = json.load(json_file)

    with open(args.reference, 'r') as json_file:
        refs_list = json.load(json_file)

    scores = compute_metric_inference(
        gens_list=gens_list,
        refs_list=refs_list,
        calculate_diversity=args.annotation is not None,
        train_file=args.annotation)
    logger.info(scores)


def parse_args():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=str, required=True,
                        help='path to the generation file')
    parser.add_argument('--reference', type=str, required=True,
                        help='path to the reference file')
    parser.add_argument('--annotation', type=str, required=False,
                        help='path to vcg annotation. If not specified, do not compute novel and unique')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
