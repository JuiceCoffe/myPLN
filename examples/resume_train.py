import argparse
import logging as log
import time
from pprint import pformat

import sys

sys.path.insert(0, ".")

import vedanet as vn
from utils.envs import initEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume training from a saved checkpoint")
    parser.add_argument("model_name", help="model name")
    parser.add_argument("--weights", required=True, help="checkpoint path to resume from")
    args = parser.parse_args()

    train_flag = 1
    config = initEnv(train_flag=train_flag, model_name=args.model_name)
    config["weights"] = args.weights

    log.info("Config\n\n%s\n", pformat(config))

    hyper_params = vn.hyperparams.HyperParams(config, train_flag=train_flag)
    eng = vn.engine.PLNTrainingEngine(hyper_params) if hyper_params.task == "pln" else vn.engine.VOCTrainingEngine(hyper_params)

    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch

    if b2 > b1:
        log.info("\nDuration of %d batches: %s seconds [%.3f sec/batch]", b2 - b1, t2 - t1, (t2 - t1) / (b2 - b1))
    else:
        log.info("\nDuration: %s seconds", t2 - t1)
