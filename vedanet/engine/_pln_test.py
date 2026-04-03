import logging as log

import torch

from .. import data
from .. import models
from ._pln_utils import collect_pln_detections, evaluate_map, write_voc_results

__all__ = ["PLNTest"]


def PLNTest(hyper_params):
    net = models.__dict__[hyper_params.model_name](
        hyper_params.classes,
        hyper_params.weights,
        train_flag=2,
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

    loader = torch.utils.data.DataLoader(
        data.PLNTestDataset(
            list_file=hyper_params.test_list,
            image_root=hyper_params.image_root,
            path_remap=hyper_params.path_remap,
            input_dimension=hyper_params.network_size,
        ),
        batch_size=hyper_params.batch,
        shuffle=False,
        drop_last=False,
        num_workers=hyper_params.nworkers if hyper_params.cuda else 0,
        pin_memory=hyper_params.pin_mem if hyper_params.cuda else False,
        collate_fn=data.pln_test_collate,
    )

    detections_by_image = {}
    gt_by_image = {}

    for images, targets, image_paths in loader:
        if hyper_params.cuda:
            images = images.cuda(non_blocking=True)
        with torch.no_grad():
            outputs = net._forward(images)

        detections_by_image.update(
            collect_pln_detections(
                outputs,
                image_paths,
                p_threshold=hyper_params.p_threshold,
                score_threshold=hyper_params.score_threshold,
                nms_score_threshold=hyper_params.nms_score,
                iou_threshold=hyper_params.nms_thresh,
                min_size=hyper_params.min_size,
                aspect_ratio_threshold=hyper_params.aspect_ratio_threshold,
                overlap_threshold=hyper_params.overlap_threshold,
                center_dist_threshold=hyper_params.center_dist_threshold,
                area_ratio_threshold=hyper_params.area_ratio_threshold,
                pre_nms_topk=hyper_params.pre_nms_topk,
                max_detections=hyper_params.max_detections,
                grid_size=hyper_params.grid_size,
                num_classes=hyper_params.classes,
            )
        )
        for batch_idx, image_path in enumerate(image_paths):
            gt_by_image[image_path] = targets[batch_idx]

    write_voc_results(
        hyper_params.results_dir,
        detections_by_image,
        labels=hyper_params.labels,
        input_dimension=hyper_params.network_size,
    )
    metrics = evaluate_map(
        detections_by_image,
        gt_by_image,
        num_classes=hyper_params.classes,
        iou_thresholds=hyper_params.eval_iou_thresholds,
        labels=hyper_params.labels,
        input_dimension=hyper_params.network_size,
    )
    log.info("PLN mAP@0.5: %.4f", metrics.get("mAP@0.5", 0.0))
    for class_idx, ap in metrics.get("per_class_ap50", {}).items():
        log.info("%s AP@0.5: %.4f", hyper_params.labels[class_idx], ap)
    return metrics
