import argparse
import json
import os
import time
from pprint import pformat

import logging as log

import sys

sys.path.insert(0, ".")

import vedanet as vn
from utils.envs import initEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PLN eval in a separate process and dump metrics to JSON")
    parser.add_argument("model_name", help="model name")
    parser.add_argument("--weights", required=True, help="checkpoint to evaluate")
    parser.add_argument("--output-json", required=True, help="json file for eval metrics")
    parser.add_argument("--batch", type=int, required=True, help="training batch associated with this checkpoint")
    parser.add_argument("--batch-size", type=int, default=None, help="eval batch size override")
    parser.add_argument("--test-list", default=None, help="eval list override")
    parser.add_argument("--results-dir", default=None, help="directory to write VOC style outputs")
    parser.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES override")
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    train_flag = 2
    config = initEnv(train_flag=train_flag, model_name=args.model_name)
    config["weights"] = args.weights
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.test_list is not None:
        config["test_list"] = args.test_list
    if args.results_dir is not None:
        config["results"] = args.results_dir

    log.info("Async Eval Config\n\n%s\n", pformat(config))

    hyper_params = vn.hyperparams.HyperParams(config, train_flag=train_flag)
    start = time.time()
    metrics = vn.engine.PLNTest(hyper_params) if hyper_params.task == "pln" else {}
    elapsed = time.time() - start

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "batch": args.batch,
                "weights": args.weights,
                "elapsed_seconds": elapsed,
                "metric_name": hyper_params.eval_metric_name,
                "metric_value": metrics.get(hyper_params.eval_metric_name, 0.0),
                "metrics": metrics,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
