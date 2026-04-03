import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

_BRANCH_CACHE = {}


def branch_search_area(branch, center_x, center_y, grid_size):
    if branch == 0:
        return range(0, center_x + 1), range(center_y, grid_size)
    if branch == 1:
        return range(0, center_x + 1), range(0, center_y + 1)
    if branch == 2:
        return range(center_x, grid_size), range(center_y, grid_size)
    return range(center_x, grid_size), range(0, center_y + 1)


def restore_bbox(branch, center_x, center_y, corner_x, corner_y):
    if branch == 0:
        bbox = [corner_x, (2 * center_y) - corner_y, (2 * center_x) - corner_x, corner_y]
    elif branch == 1:
        bbox = [corner_x, corner_y, (2 * center_x) - corner_x, (2 * center_y) - corner_y]
    elif branch == 2:
        bbox = [(2 * center_x) - corner_x, (2 * center_y) - corner_y, corner_x, corner_y]
    else:
        bbox = [(2 * center_x) - corner_x, corner_y, corner_x, (2 * center_y) - corner_y]
    return [coord * 32.0 for coord in bbox]


def restore_bbox_tensor(branch, center_x, center_y, corner_x, corner_y):
    if branch == 0:
        parts = [corner_x, (2 * center_y) - corner_y, (2 * center_x) - corner_x, corner_y]
    elif branch == 1:
        parts = [corner_x, corner_y, (2 * center_x) - corner_x, (2 * center_y) - corner_y]
    elif branch == 2:
        parts = [(2 * center_x) - corner_x, (2 * center_y) - corner_y, corner_x, corner_y]
    else:
        parts = [(2 * center_x) - corner_x, corner_y, corner_x, (2 * center_y) - corner_y]
    return torch.stack(parts, dim=1) * 32.0


def get_branch_tensors(grid_size, branch, device):
    key = (grid_size, branch)
    if key not in _BRANCH_CACHE:
        rows = torch.arange(grid_size, dtype=torch.long)
        cols = torch.arange(grid_size, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        flat_rows = grid_y.reshape(-1)
        flat_cols = grid_x.reshape(-1)

        center_rows = flat_rows[:, None]
        center_cols = flat_cols[:, None]
        corner_rows = flat_rows[None, :]
        corner_cols = flat_cols[None, :]
        if branch == 0:
            allow = (corner_cols <= center_cols) & (corner_rows >= center_rows)
        elif branch == 1:
            allow = (corner_cols <= center_cols) & (corner_rows <= center_rows)
        elif branch == 2:
            allow = (corner_cols >= center_cols) & (corner_rows >= center_rows)
        else:
            allow = (corner_cols >= center_cols) & (corner_rows <= center_rows)

        _BRANCH_CACHE[key] = (flat_rows, flat_cols, allow)

    rows, cols, allow = _BRANCH_CACHE[key]
    return rows.to(device), cols.to(device), allow.to(device)


def decode_branch(output, branch, p_threshold=0.1, score_threshold=0.1, grid_size=14, num_classes=20, feature_size=None):
    feature_size = 1 + 2 + (2 * grid_size) + num_classes if feature_size is None else feature_size
    class_offset = 3 + (2 * grid_size)
    result = output.permute(1, 2, 0).reshape(-1, output.shape[0])
    device = result.device
    rows, cols, allow_matrix = get_branch_tensors(grid_size, branch, device)
    detections = []

    for pair_idx in range(2):
        center_offset = pair_idx * feature_size
        corner_offset = (pair_idx + 2) * feature_size

        center_conf = result[:, center_offset]
        corner_conf = result[:, corner_offset]
        valid_centers = torch.nonzero(center_conf >= p_threshold, as_tuple=False).flatten()
        valid_corners = torch.nonzero(corner_conf >= p_threshold, as_tuple=False).flatten()
        if valid_centers.numel() == 0 or valid_corners.numel() == 0:
            continue

        allowed = allow_matrix.index_select(0, valid_centers).index_select(1, valid_corners)
        if not allowed.any():
            continue

        center_rows = rows.index_select(0, valid_centers)
        center_cols = cols.index_select(0, valid_centers)
        corner_rows = rows.index_select(0, valid_corners)
        corner_cols = cols.index_select(0, valid_corners)

        center_x = center_cols.float() + result.index_select(0, valid_centers)[:, center_offset + 1]
        center_y = center_rows.float() + result.index_select(0, valid_centers)[:, center_offset + 2]
        corner_x = corner_cols.float() + result.index_select(0, valid_corners)[:, corner_offset + 1]
        corner_y = corner_rows.float() + result.index_select(0, valid_corners)[:, corner_offset + 2]

        center_cls = result.index_select(0, valid_centers)[:, center_offset + class_offset:center_offset + class_offset + num_classes]
        corner_cls = result.index_select(0, valid_corners)[:, corner_offset + class_offset:corner_offset + class_offset + num_classes]

        center_link_x = result.index_select(0, valid_centers)[:, center_offset + 3:center_offset + 3 + grid_size]
        center_link_y = result.index_select(0, valid_centers)[:, center_offset + 3 + grid_size:center_offset + 3 + (2 * grid_size)]
        corner_link_x = result.index_select(0, valid_corners)[:, corner_offset + 3:corner_offset + 3 + grid_size]
        corner_link_y = result.index_select(0, valid_corners)[:, corner_offset + 3 + grid_size:corner_offset + 3 + (2 * grid_size)]

        term1 = center_link_x[:, corner_cols] * center_link_y[:, corner_rows]
        term2 = corner_link_x[:, center_cols].transpose(0, 1) * corner_link_y[:, center_rows].transpose(0, 1)
        common = center_conf.index_select(0, valid_centers)[:, None] * corner_conf.index_select(0, valid_corners)[None, :]
        common = common * ((term1 + term2) * 0.5) * allowed.float()
        if torch.count_nonzero(common) == 0:
            continue

        class_scores = common[:, :, None] * center_cls[:, None, :] * corner_cls[None, :, :]
        match_idx = torch.nonzero(class_scores >= score_threshold, as_tuple=False)
        if match_idx.numel() == 0:
            continue

        center_idx = match_idx[:, 0]
        corner_idx = match_idx[:, 1]
        class_idx = match_idx[:, 2]
        boxes = restore_bbox_tensor(branch, center_x[center_idx], center_y[center_idx], corner_x[corner_idx], corner_y[corner_idx])
        scores = class_scores[center_idx, corner_idx, class_idx].unsqueeze(1)
        labels = class_idx.float().unsqueeze(1)
        detections.append(torch.cat((boxes, scores, labels), dim=1))

    if not detections:
        return torch.zeros((0, 6), dtype=torch.float32, device=device)
    return torch.cat(detections, dim=0)


def bbox_iou(box, boxes):
    xx1 = torch.maximum(box[0], boxes[:, 0])
    yy1 = torch.maximum(box[1], boxes[:, 1])
    xx2 = torch.minimum(box[2], boxes[:, 2])
    yy2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = torch.clamp(xx2 - xx1, min=0)
    inter_h = torch.clamp(yy2 - yy1, min=0)
    inter = inter_w * inter_h

    area1 = torch.clamp(box[2] - box[0], min=0) * torch.clamp(box[3] - box[1], min=0)
    area2 = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0) * torch.clamp(boxes[:, 3] - boxes[:, 1], min=0)
    union = area1 + area2 - inter + 1e-9
    return inter / union


def filter_detections(detections, score_threshold=0.1, min_size=0.0, aspect_ratio_threshold=100.0):
    if detections.numel() == 0:
        return detections

    widths = detections[:, 2] - detections[:, 0]
    heights = detections[:, 3] - detections[:, 1]
    areas = widths * heights
    valid_geom = (widths > 0) & (heights > 0)
    aspect = torch.maximum(widths / (heights + 1e-6), heights / (widths + 1e-6))
    valid = (
        (detections[:, 4] >= score_threshold)
        & valid_geom
        & (areas >= min_size)
        & (aspect <= aspect_ratio_threshold)
    )
    return detections[valid]


def advanced_nms_by_class(
    detections,
    nms_score_threshold=0.1,
    iou_threshold=0.5,
    min_size=0.0,
    aspect_ratio_threshold=100.0,
    overlap_threshold=1.5,
    center_dist_threshold=0.0,
    area_ratio_threshold=0.5,
    pre_nms_topk=512,
    max_detections=0,
):
    detections = filter_detections(
        detections,
        score_threshold=nms_score_threshold,
        min_size=min_size,
        aspect_ratio_threshold=aspect_ratio_threshold,
    )
    if detections.numel() == 0:
        return detections

    kept = []
    for class_idx in detections[:, 5].unique(sorted=True):
        cls_boxes = detections[detections[:, 5] == class_idx]
        order = torch.argsort(cls_boxes[:, 4], descending=True)
        cls_boxes = cls_boxes[order]
        if pre_nms_topk > 0 and cls_boxes.shape[0] > pre_nms_topk:
            cls_boxes = cls_boxes[:pre_nms_topk]

        num_boxes = cls_boxes.shape[0]
        if num_boxes == 0:
            continue
        if num_boxes == 1:
            kept.append(cls_boxes)
            continue

        device = cls_boxes.device
        areas = (cls_boxes[:, 2] - cls_boxes[:, 0]) * (cls_boxes[:, 3] - cls_boxes[:, 1])
        centers_x = (cls_boxes[:, 0] + cls_boxes[:, 2]) * 0.5
        centers_y = (cls_boxes[:, 1] + cls_boxes[:, 3]) * 0.5
        widths = cls_boxes[:, 2] - cls_boxes[:, 0]
        heights = cls_boxes[:, 3] - cls_boxes[:, 1]
        eps = 1e-6

        x1 = cls_boxes[:, 0][:, None]
        y1 = cls_boxes[:, 1][:, None]
        x2 = cls_boxes[:, 2][:, None]
        y2 = cls_boxes[:, 3][:, None]

        inter_x1 = torch.maximum(x1, x1.transpose(0, 1))
        inter_y1 = torch.maximum(y1, y1.transpose(0, 1))
        inter_x2 = torch.minimum(x2, x2.transpose(0, 1))
        inter_y2 = torch.minimum(y2, y2.transpose(0, 1))
        inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        union = areas[:, None] + areas[None, :] - inter + eps
        suppress_matrix = inter / union >= iou_threshold

        if center_dist_threshold > 0.0 or overlap_threshold < float("inf"):
            norm_w = torch.maximum(widths[:, None], widths[None, :]) + eps
            norm_h = torch.maximum(heights[:, None], heights[None, :]) + eps
            center_dist = torch.sqrt(
                ((centers_x[:, None] - centers_x[None, :]) / norm_w) ** 2
                + ((centers_y[:, None] - centers_y[None, :]) / norm_h) ** 2
            )
            min_area = torch.minimum(areas[:, None], areas[None, :])
            max_area = torch.maximum(areas[:, None], areas[None, :])
            area_ratio = min_area / (max_area + eps)

            if center_dist_threshold > 0.0:
                suppress_matrix |= (center_dist < center_dist_threshold) & (area_ratio < area_ratio_threshold)
            if overlap_threshold < float("inf"):
                overlap_ratio = inter / (min_area + eps)
                suppress_matrix |= overlap_ratio > overlap_threshold

        suppress_matrix = torch.triu(suppress_matrix, diagonal=1)
        suppressed = torch.zeros(num_boxes, dtype=torch.bool, device=device)
        keep_indices = []
        for idx in range(num_boxes):
            if suppressed[idx]:
                continue
            keep_indices.append(idx)
            suppressed |= suppress_matrix[idx]

        if keep_indices:
            kept.append(cls_boxes[torch.tensor(keep_indices, device=device, dtype=torch.long)])

    if not kept:
        return detections.new_zeros((0, 6))

    merged = torch.cat(kept, dim=0)
    order = torch.argsort(merged[:, 4], descending=True)
    merged = merged[order]
    if max_detections > 0 and merged.shape[0] > max_detections:
        merged = merged[:max_detections]
    return merged


def collect_pln_detections(
    outputs,
    image_paths,
    p_threshold=0.1,
    score_threshold=0.1,
    nms_score_threshold=0.1,
    iou_threshold=0.5,
    min_size=0.0,
    aspect_ratio_threshold=100.0,
    overlap_threshold=1.5,
    center_dist_threshold=0.0,
    area_ratio_threshold=0.5,
    pre_nms_topk=512,
    max_detections=0,
    grid_size=14,
    num_classes=20,
):
    detections_by_image = {}
    batch_size = outputs[0].shape[0]

    for batch_idx in range(batch_size):
        per_branch = []
        for branch_idx in range(4):
            dets = decode_branch(
                outputs[branch_idx][batch_idx],
                branch=branch_idx,
                p_threshold=p_threshold,
                score_threshold=score_threshold,
                grid_size=grid_size,
                num_classes=num_classes,
            )
            if dets.numel() > 0:
                per_branch.append(dets)

        merged = torch.cat(per_branch, dim=0) if per_branch else torch.zeros((0, 6), dtype=torch.float32)
        merged = advanced_nms_by_class(
            merged,
            nms_score_threshold=nms_score_threshold,
            iou_threshold=iou_threshold,
            min_size=min_size,
            aspect_ratio_threshold=aspect_ratio_threshold,
            overlap_threshold=overlap_threshold,
            center_dist_threshold=center_dist_threshold,
            area_ratio_threshold=area_ratio_threshold,
            pre_nms_topk=pre_nms_topk,
            max_detections=max_detections,
        )
        detections_by_image[image_paths[batch_idx]] = merged.detach().cpu()

    return detections_by_image


def _voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.0
        for threshold in np.arange(0.0, 1.1, 0.1):
            mask = rec >= threshold
            ap += (np.max(prec[mask]) if np.any(mask) else 0.0) / 11.0
        return ap

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = np.maximum(mpre[idx - 1], mpre[idx])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1])


def _normalize_input_dimension(input_dimension):
    if input_dimension is None:
        return None
    if isinstance(input_dimension, (int, float)):
        size = float(input_dimension)
        return size, size
    if len(input_dimension) != 2:
        return None
    return float(input_dimension[0]), float(input_dimension[1])


def _infer_voc_annotation_path(image_path):
    token = f"{os.sep}JPEGImages{os.sep}"
    normalized_path = os.path.normpath(image_path)
    if token not in normalized_path:
        return None
    annotation_path = normalized_path.replace(token, f"{os.sep}Annotations{os.sep}", 1)
    return os.path.splitext(annotation_path)[0] + ".xml"


def _extract_voc_year(path):
    match = re.search(r"VOC(\d{4})", path)
    return int(match.group(1)) if match else None


def _parse_voc_annotation(annotation_path, label_to_index):
    try:
        root = ET.parse(annotation_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return None

    width = root.findtext("size/width")
    height = root.findtext("size/height")
    if width is None or height is None:
        return None

    objects = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        if class_name not in label_to_index:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        objects.append(
            {
                "bbox": np.array(
                    [
                        float(bbox.findtext("xmin")),
                        float(bbox.findtext("ymin")),
                        float(bbox.findtext("xmax")),
                        float(bbox.findtext("ymax")),
                    ],
                    dtype=np.float64,
                ),
                "difficult": bool(int(obj.findtext("difficult", default="0"))),
                "label": label_to_index[class_name],
            }
        )

    return {"width": float(width), "height": float(height), "objects": objects}


def _prepare_voc_records(detections_by_image, labels):
    label_to_index = {name: idx for idx, name in enumerate(labels)}
    years = set()
    records = {}

    for image_path in detections_by_image:
        annotation_path = _infer_voc_annotation_path(image_path)
        if annotation_path is None or not os.path.exists(annotation_path):
            return None
        record = _parse_voc_annotation(annotation_path, label_to_index)
        if record is None:
            return None
        records[image_path] = record
        year = _extract_voc_year(annotation_path) or _extract_voc_year(image_path)
        if year is not None:
            years.add(year)

    use_07_metric = bool(years) and max(years) < 2010
    return records, use_07_metric


def _scale_detections_to_original(detections_by_image, annotation_records, input_dimension):
    netw, neth = input_dimension
    scaled_detections = {}

    for image_path, detections in detections_by_image.items():
        if detections.numel() == 0:
            scaled_detections[image_path] = np.zeros((0, 6), dtype=np.float64)
            continue

        record = annotation_records[image_path]
        scaled = detections.detach().cpu().numpy().astype(np.float64, copy=True)
        width = record["width"]
        height = record["height"]
        scaled[:, [0, 2]] *= width / netw
        scaled[:, [1, 3]] *= height / neth
        scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0.0, max(width - 1.0, 0.0))
        scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0.0, max(height - 1.0, 0.0))
        scaled_detections[image_path] = scaled

    return scaled_detections


def _summarize_ap(ap_by_threshold):
    summary = {}
    for iou_threshold, aps in ap_by_threshold.items():
        summary[f"mAP@{iou_threshold}"] = sum(aps.values()) / max(len(aps), 1)

    if 0.5 in ap_by_threshold:
        summary["per_class_ap50"] = ap_by_threshold[0.5]
    return summary


def _evaluate_map_legacy(detections_by_image, gt_by_image, num_classes=20, iou_thresholds=(0.5,)):
    ap_by_threshold = {}
    for iou_threshold in iou_thresholds:
        aps = {}
        for class_idx in range(num_classes):
            class_dets = []
            gt_count = 0
            gt_used = {}

            for image_id, gts in gt_by_image.items():
                cls_gts = gts[gts[:, 4] == class_idx]
                gt_count += cls_gts.shape[0]
                gt_used[image_id] = torch.zeros(cls_gts.shape[0], dtype=torch.bool)

            for image_id, dets in detections_by_image.items():
                cls_dets = dets[dets[:, 5] == class_idx]
                for det in cls_dets:
                    class_dets.append((image_id, float(det[4]), det[:4]))

            if gt_count == 0:
                aps[class_idx] = 0.0
                continue

            class_dets.sort(key=lambda item: item[1], reverse=True)
            tp = torch.zeros(len(class_dets), dtype=torch.float32)
            fp = torch.zeros(len(class_dets), dtype=torch.float32)

            for idx, (image_id, _, det_box) in enumerate(class_dets):
                gts = gt_by_image.get(image_id, torch.zeros((0, 5), dtype=torch.float32))
                cls_gts = gts[gts[:, 4] == class_idx]
                if cls_gts.numel() == 0:
                    fp[idx] = 1.0
                    continue

                ious = bbox_iou(det_box, cls_gts[:, :4])
                max_iou, max_idx = torch.max(ious, dim=0)
                if max_iou >= iou_threshold and not gt_used[image_id][max_idx]:
                    tp[idx] = 1.0
                    gt_used[image_id][max_idx] = True
                else:
                    fp[idx] = 1.0

            cum_tp = torch.cumsum(tp, dim=0)
            cum_fp = torch.cumsum(fp, dim=0)
            recall = cum_tp / max(gt_count, 1)
            precision = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-9)

            ap = 0.0
            for threshold in torch.arange(0.0, 1.1, 0.1):
                mask = recall >= threshold
                ap += float(torch.max(precision[mask])) if mask.any() else 0.0
            aps[class_idx] = ap / 11.0

        ap_by_threshold[iou_threshold] = aps

    return _summarize_ap(ap_by_threshold)


def _evaluate_map_voc(detections_by_image, annotation_records, use_07_metric, num_classes=20, iou_thresholds=(0.5,), input_dimension=(448.0, 448.0)):
    scaled_detections = _scale_detections_to_original(detections_by_image, annotation_records, input_dimension)
    ap_by_threshold = {}

    for iou_threshold in iou_thresholds:
        aps = {}
        for class_idx in range(num_classes):
            class_recs = {}
            gt_count = 0
            for image_path, record in annotation_records.items():
                class_objects = [obj for obj in record["objects"] if obj["label"] == class_idx]
                if class_objects:
                    boxes = np.array([obj["bbox"] for obj in class_objects], dtype=np.float64)
                    difficult = np.array([obj["difficult"] for obj in class_objects], dtype=bool)
                else:
                    boxes = np.zeros((0, 4), dtype=np.float64)
                    difficult = np.zeros((0,), dtype=bool)
                class_recs[image_path] = {"bbox": boxes, "difficult": difficult, "det": [False] * len(class_objects)}
                gt_count += int(np.sum(~difficult))

            if gt_count == 0:
                aps[class_idx] = 0.0
                continue

            image_ids = []
            confidence = []
            boxes = []
            for image_path, detections in scaled_detections.items():
                class_detections = detections[detections[:, 5] == class_idx]
                for detection in class_detections:
                    image_ids.append(image_path)
                    confidence.append(float(detection[4]))
                    boxes.append(detection[:4])

            if not confidence:
                aps[class_idx] = 0.0
                continue

            confidence = np.array(confidence, dtype=np.float64)
            boxes = np.array(boxes, dtype=np.float64)
            sorted_indices = np.argsort(-confidence)
            boxes = boxes[sorted_indices, :]
            image_ids = [image_ids[idx] for idx in sorted_indices]

            tp = np.zeros(len(image_ids), dtype=np.float64)
            fp = np.zeros(len(image_ids), dtype=np.float64)

            for det_idx, image_path in enumerate(image_ids):
                record = class_recs[image_path]
                det_box = boxes[det_idx, :]
                gt_boxes = record["bbox"]
                ovmax = -np.inf
                match_idx = -1

                if gt_boxes.size > 0:
                    ixmin = np.maximum(gt_boxes[:, 0], det_box[0])
                    iymin = np.maximum(gt_boxes[:, 1], det_box[1])
                    ixmax = np.minimum(gt_boxes[:, 2], det_box[2])
                    iymax = np.minimum(gt_boxes[:, 3], det_box[3])
                    inter_w = np.maximum(ixmax - ixmin + 1.0, 0.0)
                    inter_h = np.maximum(iymax - iymin + 1.0, 0.0)
                    intersections = inter_w * inter_h
                    unions = (
                        (det_box[2] - det_box[0] + 1.0) * (det_box[3] - det_box[1] + 1.0)
                        + (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0)
                        - intersections
                    )
                    overlaps = intersections / np.maximum(unions, np.finfo(np.float64).eps)
                    ovmax = np.max(overlaps)
                    match_idx = int(np.argmax(overlaps))

                if ovmax > iou_threshold:
                    if not record["difficult"][match_idx]:
                        if not record["det"][match_idx]:
                            tp[det_idx] = 1.0
                            record["det"][match_idx] = True
                        else:
                            fp[det_idx] = 1.0
                else:
                    fp[det_idx] = 1.0

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            recall = tp / float(gt_count)
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            aps[class_idx] = float(_voc_ap(recall, precision, use_07_metric=use_07_metric))

        ap_by_threshold[iou_threshold] = aps

    return _summarize_ap(ap_by_threshold)


def evaluate_map(
    detections_by_image,
    gt_by_image,
    num_classes=20,
    iou_thresholds=(0.5,),
    labels=VOC_CLASSES,
    input_dimension=None,
):
    normalized_input_dimension = _normalize_input_dimension(input_dimension)
    if normalized_input_dimension is not None and detections_by_image:
        prepared = _prepare_voc_records(detections_by_image, labels)
        if prepared is not None:
            annotation_records, use_07_metric = prepared
            return _evaluate_map_voc(
                detections_by_image,
                annotation_records,
                use_07_metric=use_07_metric,
                num_classes=num_classes,
                iou_thresholds=iou_thresholds,
                input_dimension=normalized_input_dimension,
            )
    return _evaluate_map_legacy(detections_by_image, gt_by_image, num_classes=num_classes, iou_thresholds=iou_thresholds)


def write_voc_results(results_dir, detections_by_image, labels=VOC_CLASSES, input_dimension=None):
    os.makedirs(results_dir, exist_ok=True)
    per_class_lines = defaultdict(list)

    normalized_input_dimension = _normalize_input_dimension(input_dimension)
    prepared = _prepare_voc_records(detections_by_image, labels) if normalized_input_dimension is not None and detections_by_image else None
    if prepared is not None:
        annotation_records, _ = prepared
        detections_to_write = _scale_detections_to_original(detections_by_image, annotation_records, normalized_input_dimension)
    else:
        detections_to_write = detections_by_image

    for image_path, detections in detections_to_write.items():
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        for det in detections:
            if isinstance(det, np.ndarray):
                class_idx = int(det[5])
                score = float(det[4])
                xmin, ymin, xmax, ymax = (float(det[idx]) for idx in range(4))
            else:
                class_idx = int(det[5].item())
                score = det[4].item()
                xmin, ymin, xmax, ymax = (det[idx].item() for idx in range(4))
            label = labels[class_idx]
            line = f"{image_id} {score:.6f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}"
            per_class_lines[label].append(line)

    for label in labels:
        out_path = os.path.join(results_dir, f"comp4_det_test_{label}.txt")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(per_class_lines[label]))
