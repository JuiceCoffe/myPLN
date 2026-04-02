import logging as log
import time

import torch

from .. import models

__all__ = ["pln_speed"]


def pln_speed(hyper_params):
    net = models.__dict__[hyper_params.model_name](
        hyper_params.classes,
        train_flag=0,
        backbone_pretrained=False,
        point_weight=hyper_params.point_weight,
        coord_weight=hyper_params.coord_weight,
        link_weight=hyper_params.link_weight,
        class_weight=hyper_params.class_weight,
        grid_size=hyper_params.grid_size,
    )
    net.eval()
    if hyper_params.cuda:
        net.cuda()

    width, height = hyper_params.network_size
    data = torch.randn(hyper_params.batch, 3, height, width, dtype=torch.float32)
    if hyper_params.cuda:
        data = data.cuda()

    if hyper_params.cuda:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(hyper_params.max_iters):
        with torch.no_grad():
            net._forward(data)
    if hyper_params.cuda:
        torch.cuda.synchronize()

    log.info("PLN speed: %.3f ms/iter", (time.time() - start) * 1000.0 / hyper_params.max_iters)
