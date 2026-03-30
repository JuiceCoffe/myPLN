import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from ._dataloading import Dataset

__all__ = ["PLNTrainDataset", "PLNTestDataset", "pln_test_collate"]


VOC_MEAN = np.array((123.0, 117.0, 104.0), dtype=np.float32)


def _to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def _resolve_path(root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(root, path) if root else path


class PLNAnnotationParser:
    def __init__(self, root_dir=""):
        self.root_dir = root_dir

    def parse(self, list_file):
        samples = []
        with open(list_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                image_path = _resolve_path(self.root_dir, parts[0])
                boxes = []
                labels = []
                for idx in range((len(parts) - 1) // 5):
                    start = 1 + (idx * 5)
                    boxes.append([
                        float(parts[start]),
                        float(parts[start + 1]),
                        float(parts[start + 2]),
                        float(parts[start + 3]),
                    ])
                    labels.append(int(parts[start + 4]))
                samples.append((image_path, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)))
        return samples


class PLNLabelGenerator:
    def __init__(self, grid_size=14, num_classes=20, num_pairs=2):
        self.s = grid_size
        self.num_classes = num_classes
        self.num_pairs = num_pairs
        self.large_value = 1e8

    def generate(self, branch, boxes, labels):
        if boxes.numel() == 0:
            channels = (1 + 2 + (2 * self.s) + self.num_classes) * 4
            return torch.zeros((self.s, self.s, channels), dtype=torch.float32)

        q_corner, q_center = self._generate_class_tensors(branch, boxes, labels)
        p_corner, p_center = self._generate_point_tensors(branch, boxes)
        link_center, link_corner = self._generate_link_tensors(branch, boxes)
        pos_corner, pos_center = self._generate_position_tensors(branch, boxes)

        corner_features = []
        center_features = []
        for idx in range(self.num_pairs):
            center_features.append(torch.cat((p_center[idx], pos_center[idx], link_center[idx], q_center[idx]), dim=-1))
            corner_features.append(torch.cat((p_corner[idx], pos_corner[idx], link_corner[idx], q_corner[idx]), dim=-1))

        return torch.cat((torch.cat(center_features, dim=-1), torch.cat(corner_features, dim=-1)), dim=-1)

    def _clamp_box(self, box):
        xmin, ymin, xmax, ymax = box.tolist()
        xmax = min(xmax, 0.999)
        ymax = min(ymax, 0.999)
        return xmin, ymin, xmax, ymax

    def _branch_coords(self, branch, xmin, ymin, xmax, ymax):
        if branch == 0:
            return xmin * self.s, ymax * self.s
        if branch == 1:
            return xmin * self.s, ymin * self.s
        if branch == 2:
            return xmax * self.s, ymax * self.s
        return xmax * self.s, ymin * self.s

    def _branch_indices(self, branch, xmin, ymin, xmax, ymax):
        xf, yf = self._branch_coords(branch, xmin, ymin, xmax, ymax)
        return int(yf), int(xf)

    def _generate_point_tensors(self, branch, boxes):
        corner_list = [torch.zeros((self.s, self.s, 1), dtype=torch.float32) for _ in range(self.num_pairs)]
        center_list = [torch.zeros((self.s, self.s, 1), dtype=torch.float32) for _ in range(self.num_pairs)]

        for box in boxes:
            xmin, ymin, xmax, ymax = self._clamp_box(box)
            corner_y, corner_x = self._branch_indices(branch, xmin, ymin, xmax, ymax)
            center_y = int(((ymin + ymax) * 0.5) * self.s)
            center_x = int(((xmin + xmax) * 0.5) * self.s)
            for idx in range(self.num_pairs):
                corner_list[idx][corner_y, corner_x, 0] = 1.0
                center_list[idx][center_y, center_x, 0] = 1.0
        return corner_list, center_list

    def _generate_position_tensors(self, branch, boxes):
        corner_list = [torch.zeros((self.s, self.s, 2), dtype=torch.float32) for _ in range(self.num_pairs)]
        center_list = [torch.zeros((self.s, self.s, 2), dtype=torch.float32) for _ in range(self.num_pairs)]

        for box in boxes:
            xmin, ymin, xmax, ymax = self._clamp_box(box)
            corner_y, corner_x = self._branch_indices(branch, xmin, ymin, xmax, ymax)
            corner_xf, corner_yf = self._branch_coords(branch, xmin, ymin, xmax, ymax)
            corner_offset = torch.tensor([corner_xf - corner_x, corner_yf - corner_y], dtype=torch.float32)

            center_xf = ((xmin + xmax) * 0.5) * self.s
            center_yf = ((ymin + ymax) * 0.5) * self.s
            center_x = int(center_xf)
            center_y = int(center_yf)
            center_offset = torch.tensor([center_xf - center_x, center_yf - center_y], dtype=torch.float32)

            for idx in range(self.num_pairs):
                corner_list[idx][corner_y, corner_x] = corner_offset
                center_list[idx][center_y, center_x] = center_offset
        return corner_list, center_list

    def _generate_class_tensors(self, branch, boxes, labels):
        corner_list = [torch.zeros((self.s, self.s, self.num_classes), dtype=torch.float32) for _ in range(self.num_pairs)]
        center_list = [torch.zeros((self.s, self.s, self.num_classes), dtype=torch.float32) for _ in range(self.num_pairs)]

        for box, class_label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = self._clamp_box(box)
            corner_y, corner_x = self._branch_indices(branch, xmin, ymin, xmax, ymax)
            center_y = int(((ymin + ymax) * 0.5) * self.s)
            center_x = int(((xmin + xmax) * 0.5) * self.s)
            for idx in range(self.num_pairs):
                corner_list[idx][corner_y, corner_x, class_label] = self.large_value
                center_list[idx][center_y, center_x, class_label] = self.large_value

        corner_list = [F.softmax(item, dim=-1) for item in corner_list]
        center_list = [F.softmax(item, dim=-1) for item in center_list]
        return corner_list, center_list

    def _generate_link_tensors(self, branch, boxes):
        center_list = [torch.zeros((self.s, self.s, 2 * self.s), dtype=torch.float32) for _ in range(self.num_pairs)]
        corner_list = [torch.zeros((self.s, self.s, 2 * self.s), dtype=torch.float32) for _ in range(self.num_pairs)]

        for box in boxes:
            xmin, ymin, xmax, ymax = self._clamp_box(box)
            center_x = ((xmin + xmax) * 0.5) * self.s
            center_y = ((ymin + ymax) * 0.5) * self.s
            corner_xf, corner_yf = self._branch_coords(branch, xmin, ymin, xmax, ymax)
            center_ix, center_iy = int(center_x), int(center_y)
            corner_ix, corner_iy = int(corner_xf), int(corner_yf)

            center_link = torch.zeros(2 * self.s, dtype=torch.float32)
            center_link[corner_ix] = self.large_value
            center_link[self.s + corner_iy] = self.large_value

            corner_link = torch.zeros(2 * self.s, dtype=torch.float32)
            corner_link[center_ix] = self.large_value
            corner_link[self.s + center_iy] = self.large_value

            for idx in range(self.num_pairs):
                center_list[idx][center_iy, center_ix] = center_link
                corner_list[idx][corner_iy, corner_ix] = corner_link

        for idx in range(self.num_pairs):
            center_list[idx][..., :self.s] = F.softmax(center_list[idx][..., :self.s], dim=-1)
            center_list[idx][..., self.s:] = F.softmax(center_list[idx][..., self.s:], dim=-1)
            corner_list[idx][..., :self.s] = F.softmax(corner_list[idx][..., :self.s], dim=-1)
            corner_list[idx][..., self.s:] = F.softmax(corner_list[idx][..., self.s:], dim=-1)
        return center_list, corner_list


class PLNTrainDataset(Dataset):
    def __init__(self, list_file, image_root="", input_dimension=(448, 448), grid_size=14, num_classes=20, flip=0.0):
        super().__init__(input_dimension)
        self.samples = PLNAnnotationParser(image_root).parse(list_file)
        self.flip = flip
        self.generator = PLNLabelGenerator(grid_size=grid_size, num_classes=num_classes)

    def __len__(self):
        return len(self.samples)

    @Dataset.resize_getitem
    def __getitem__(self, index):
        image_path, boxes, labels = self.samples[index]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image [{image_path}]")

        height, width = image.shape[:2]
        boxes = boxes.clone()
        labels = labels.clone()

        if self.flip > 0 and random.random() < self.flip and boxes.numel() > 0:
            image = cv2.flip(image, 1)
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = width - x2
            boxes[:, 2] = width - x1

        norm_boxes = boxes / torch.tensor([width, height, width, height], dtype=torch.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image -= VOC_MEAN
        image = cv2.resize(image, tuple(self.input_dim))
        image = _to_tensor(image)

        targets = [self.generator.generate(branch, norm_boxes, labels) for branch in range(4)]
        target_tensor = torch.stack(targets)
        return image, target_tensor


class PLNTestDataset(Dataset):
    def __init__(self, list_file, image_root="", input_dimension=(448, 448)):
        super().__init__(input_dimension)
        self.samples = PLNAnnotationParser(image_root).parse(list_file)

    def __len__(self):
        return len(self.samples)

    @Dataset.resize_getitem
    def __getitem__(self, index):
        image_path, boxes, labels = self.samples[index]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image [{image_path}]")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image -= VOC_MEAN
        image = cv2.resize(image, tuple(self.input_dim))
        image = _to_tensor(image)

        target = torch.zeros((boxes.shape[0], 5), dtype=torch.float32)
        if boxes.numel() > 0:
            target[:, :4] = boxes
            target[:, 4] = labels.float()
        return image, target, image_path


def pln_test_collate(batch):
    images, targets, image_paths = zip(*batch)
    return default_collate(images), list(targets), list(image_paths)
