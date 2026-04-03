import logging as log
import math
import json
import os
import shutil
from statistics import mean
import subprocess
import sys

import torch

from .. import data
from .. import models
from . import engine
from ._pln_utils import collect_pln_detections, evaluate_map, write_voc_results

__all__ = ["PLNTrainingEngine"]


class _AttrDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class PLNTrainingEngine(engine.Engine):
    @property
    def learning_rate(self):
        return getattr(self, '_base_learning_rate', self.optimizer.param_groups[-1]['lr'])

    @learning_rate.setter
    def learning_rate(self, lr):
        self._base_learning_rate = lr
        applied = []
        for param_group in self.optimizer.param_groups:
            lr_mult = param_group.get('lr_mult', 1.0)
            group_lr = lr * lr_mult
            param_group['lr'] = group_lr
            applied.append(f"{param_group.get('name', 'group')}={group_lr:.8f}")
        log.info('Adjusting learning rates: %s', ', '.join(applied))

    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        self.batch_size = hyper_params.batch
        self.mini_batch_size = hyper_params.mini_batch
        self.max_batches = hyper_params.max_batches
        self.cuda = hyper_params.cuda
        self.backup_dir = hyper_params.backup_dir
        self.backup_rate = hyper_params.backup
        self.test_rate = hyper_params.eval_interval if hyper_params.eval_interval > 0 else None
        self.best_metric = float("-inf")
        self.val_loader = None
        self.async_eval_enabled = bool(self.test_rate is not None and hyper_params.async_eval)
        self.async_eval_dir = os.path.join(self.backup_dir, "async_eval")
        self.async_eval_process = None
        self.async_eval_request = None
        self.async_eval_log_handle = None
        self.pending_eval_request = None

        net = models.__dict__[hyper_params.model_name](
            hyper_params.classes,
            hyper_params.weights,
            train_flag=1,
            clear=hyper_params.clear,
            backbone_pretrained=hyper_params.backbone_pretrained,
            point_weight=hyper_params.point_weight,
            coord_weight=hyper_params.coord_weight,
            link_weight=hyper_params.link_weight,
            class_weight=hyper_params.class_weight,
            grid_size=hyper_params.grid_size,
        )
        log.info("Net structure\n\n%s\n", net)
        if self.cuda:
            net.cuda()
            visible_gpus = torch.cuda.device_count()
            if visible_gpus > 1:
                net = _AttrDataParallel(net, device_ids=list(range(visible_gpus)))
                log.info("Using DataParallel across %d GPUs", visible_gpus)

        model_ref = net.module if isinstance(net, _AttrDataParallel) else net
        backbone_params = list(model_ref.backbone.parameters())
        backbone_param_ids = {id(param) for param in backbone_params}
        other_params = [param for param in model_ref.parameters() if id(param) not in backbone_param_ids]

        optim = torch.optim.SGD(
            [
                {'params': backbone_params, 'lr_mult': hyper_params.backbone_lr_mult, 'name': 'backbone'},
                {'params': other_params, 'lr_mult': 1.0, 'name': 'main'},
            ],
            lr=hyper_params.learning_rate,
            momentum=hyper_params.momentum,
            dampening=0,
            weight_decay=hyper_params.decay,
        )

        dataset = data.PLNTrainDataset(
            list_file=hyper_params.train_list,
            image_root=hyper_params.image_root,
            path_remap=hyper_params.path_remap,
            input_dimension=hyper_params.network_size,
            grid_size=hyper_params.grid_size,
            num_classes=hyper_params.classes,
            flip=hyper_params.flip,
            jitter=hyper_params.jitter,
            crop_min_area=hyper_params.crop_min_area,
            hue=hyper_params.hue,
            saturation=hyper_params.sat,
            value=hyper_params.val,
            brightness=hyper_params.brightness,
            contrast=hyper_params.contrast,
            grayscale=hyper_params.grayscale,
            blur=hyper_params.blur,
            noise_std=hyper_params.noise_std,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=hyper_params.nworkers if self.cuda else 0,
            pin_memory=hyper_params.pin_mem if self.cuda else False,
        )

        super().__init__(net, optim, dataloader)
        self._base_learning_rate = hyper_params.learning_rate
        self.learning_rate = hyper_params.learning_rate
        self.loss_log = {key: [] for key in ("total", "point", "coord", "link", "class", "noobj")}

    def start(self):
        hp = self.hyper_params
        if hp.lr_policy != "cosine_warmup":
            self.add_rate("learning_rate", hp.lr_steps, hp.lr_rates)
        self.add_rate("backup_rate", hp.bp_steps, hp.bp_rates, hp.backup)
        if self.async_eval_enabled:
            os.makedirs(self.async_eval_dir, exist_ok=True)
            log.info("Async eval enabled on GPUs [%s]", hp.async_eval_gpus)
        elif hp.eval_interval > 0:
            self.val_loader = torch.utils.data.DataLoader(
                data.PLNTestDataset(
                    list_file=hp.test_list,
                    image_root=hp.image_root,
                    path_remap=hp.path_remap,
                    input_dimension=hp.network_size,
                ),
                batch_size=hp.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=hp.eval_nworkers if self.cuda else 0,
                pin_memory=hp.eval_pin_mem if self.cuda else False,
                collate_fn=data.pln_test_collate,
            )


    def _compute_learning_rate(self, batch_idx):
        hp = self.hyper_params
        if hp.lr_policy != "cosine_warmup":
            return None

        current_iteration = max(int(batch_idx), 1)
        if hp.warmup_batches > 0 and current_iteration < hp.warmup_batches:
            return hp.learning_rate + (hp.max_learning_rate - hp.learning_rate) * (current_iteration / hp.warmup_batches)

        if hp.cosine_total_batches <= 0:
            return hp.max_learning_rate

        decay_iteration = max(current_iteration - hp.warmup_batches, 0)
        progress = min(decay_iteration / hp.cosine_total_batches, 1.0)
        return hp.min_learning_rate + 0.5 * (hp.max_learning_rate - hp.min_learning_rate) * (1.0 + math.cos(math.pi * progress))

    def _apply_learning_rate(self, batch_idx):
        lr = self._compute_learning_rate(batch_idx)
        if lr is not None and lr != self.learning_rate:
            self.learning_rate = lr

    def process_batch(self, batch):
        images, targets = batch
        if self.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        loss = self.network(images, targets)
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.sum()
        loss = loss / self.batch_size
        loss.backward()

        if isinstance(self.network, _AttrDataParallel):
            self.network.module.seen += images.size(0)

        total = point = coord = link = cls = noobj = 0.0
        for criterion in self.network.loss:
            stats = criterion.last_stats
            total += stats["total"].item() / self.mini_batch_size
            point += stats["point"].item() / self.mini_batch_size
            coord += stats["coord"].item() / self.mini_batch_size
            link += stats["link"].item() / self.mini_batch_size
            cls += stats["class"].item() / self.mini_batch_size
            noobj += stats["noobj"].item() / self.mini_batch_size

        self.loss_log["total"].append(total)
        self.loss_log["point"].append(point)
        self.loss_log["coord"].append(coord)
        self.loss_log["link"].append(link)
        self.loss_log["class"].append(cls)
        self.loss_log["noobj"].append(noobj)

    def train_batch(self):
        self._apply_learning_rate(self.batch)
        self.optimizer.step()
        self.optimizer.zero_grad()

        log.info(
            "%d # Loss: %.5f (Point: %.2f Coord: %.2f Link: %.2f Class: %.2f NoObj: %.2f)",
            self.batch,
            mean(self.loss_log["total"]),
            mean(self.loss_log["point"]),
            mean(self.loss_log["coord"]),
            mean(self.loss_log["link"]),
            mean(self.loss_log["class"]),
            mean(self.loss_log["noobj"]),
        )
        self.loss_log = {key: [] for key in self.loss_log}

        if self.batch % self.backup_rate == 0:
            self.network.save_weights(os.path.join(self.backup_dir, "latest.pt"))
            self.network.save_weights(os.path.join(self.backup_dir, f"weights_{self.batch}.pt"))
        if self.batch % 100 == 0:
            self.network.save_weights(os.path.join(self.backup_dir, "backup.pt"))
        self._poll_async_eval()

    def _create_async_eval_request(self, batch):
        snapshot_path = os.path.join(self.async_eval_dir, f"eval_{batch}.pt")
        metrics_path = os.path.join(self.async_eval_dir, f"eval_{batch}.json")
        results_dir = os.path.join(self.async_eval_dir, f"results_batch_{batch}")
        log_path = os.path.join(self.async_eval_dir, f"eval_{batch}.log")
        self.network.save_weights(snapshot_path)
        return {
            "batch": batch,
            "snapshot_path": snapshot_path,
            "metrics_path": metrics_path,
            "results_dir": results_dir,
            "log_path": log_path,
        }

    def _cleanup_async_eval_request(self, request):
        if request is None:
            return
        for key in ("snapshot_path",):
            path = request.get(key)
            if path and os.path.exists(path):
                os.remove(path)

    def _launch_async_eval(self, request):
        cmd = [
            sys.executable,
            "examples/async_eval.py",
            self.hyper_params.model_name,
            "--weights",
            request["snapshot_path"],
            "--output-json",
            request["metrics_path"],
            "--batch",
            str(request["batch"]),
            "--batch-size",
            str(self.hyper_params.eval_batch_size),
            "--test-list",
            self.hyper_params.test_list,
            "--results-dir",
            request["results_dir"],
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.hyper_params.async_eval_gpus
        self.async_eval_log_handle = open(request["log_path"], "w", encoding="utf-8")
        self.async_eval_process = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            env=env,
            stdout=self.async_eval_log_handle,
            stderr=subprocess.STDOUT,
        )
        self.async_eval_request = request
        log.info("Started async eval for batch %d using [%s]", request["batch"], request["snapshot_path"])

    def _start_next_async_eval(self):
        if self.async_eval_process is None and self.pending_eval_request is not None:
            request = self.pending_eval_request
            self.pending_eval_request = None
            self._launch_async_eval(request)

    def _poll_async_eval(self):
        if self.async_eval_process is None:
            return

        return_code = self.async_eval_process.poll()
        if return_code is None:
            return

        if self.async_eval_log_handle is not None:
            self.async_eval_log_handle.close()
            self.async_eval_log_handle = None

        request = self.async_eval_request
        self.async_eval_process = None
        self.async_eval_request = None

        if return_code != 0:
            log.warning("Async eval for batch %d exited with code %d", request["batch"], return_code)
            self._cleanup_async_eval_request(request)
            self._start_next_async_eval()
            return

        if not os.path.exists(request["metrics_path"]):
            log.warning("Async eval for batch %d finished without metrics json", request["batch"])
            self._cleanup_async_eval_request(request)
            self._start_next_async_eval()
            return

        with open(request["metrics_path"], "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        current_metric = payload.get("metric_value", 0.0)
        metric_name = payload.get("metric_name", self.hyper_params.eval_metric_name)
        elapsed_seconds = payload.get("elapsed_seconds", 0.0)
        eval_batch = payload.get("batch", request["batch"])
        log.info(
            "Async eval finished for batch %d in %.2fs: %s=%.4f",
            eval_batch,
            elapsed_seconds,
            metric_name,
            current_metric,
        )

        if current_metric > self.best_metric:
            self.best_metric = current_metric
            shutil.copyfile(request["snapshot_path"], os.path.join(self.backup_dir, "best.pt"))
            log.info("New best checkpoint saved from async eval batch %d with %s=%.4f", eval_batch, metric_name, current_metric)

        self._cleanup_async_eval_request(request)
        self._start_next_async_eval()

    def _cancel_async_eval(self):
        if self.async_eval_process is not None:
            self.async_eval_process.terminate()
            try:
                self.async_eval_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.async_eval_process.kill()
                self.async_eval_process.wait(timeout=5)
            if self.async_eval_log_handle is not None:
                self.async_eval_log_handle.close()
                self.async_eval_log_handle = None
            self._cleanup_async_eval_request(self.async_eval_request)
            self.async_eval_process = None
            self.async_eval_request = None

        if self.pending_eval_request is not None:
            self._cleanup_async_eval_request(self.pending_eval_request)
            self.pending_eval_request = None

    def _wait_for_async_eval(self):
        while self.async_eval_process is not None or self.pending_eval_request is not None:
            self._poll_async_eval()
            if self.async_eval_process is None:
                self._start_next_async_eval()
                continue
            self.async_eval_process.wait()
            self._poll_async_eval()

    def test(self):
        if self.async_eval_enabled:
            request = self._create_async_eval_request(self.batch)
            if self.async_eval_process is None:
                self._launch_async_eval(request)
            else:
                if self.pending_eval_request is not None:
                    self._cleanup_async_eval_request(self.pending_eval_request)
                self.pending_eval_request = request
                log.info(
                    "Queued async eval for batch %d; batch %d eval still running",
                    self.batch,
                    self.async_eval_request["batch"],
                )
            return

        if self.val_loader is None:
            return

        detections_by_image = {}
        gt_by_image = {}
        self.network.eval()
        for images, targets, image_paths in self.val_loader:
            if self.cuda:
                images = images.cuda(non_blocking=True)
            with torch.no_grad():
                outputs = self.network._forward(images)

            detections_by_image.update(
                collect_pln_detections(
                    outputs,
                    image_paths,
                    p_threshold=self.hyper_params.p_threshold,
                    score_threshold=self.hyper_params.score_threshold,
                    nms_score_threshold=self.hyper_params.nms_score,
                    iou_threshold=self.hyper_params.nms_thresh,
                    min_size=self.hyper_params.min_size,
                    aspect_ratio_threshold=self.hyper_params.aspect_ratio_threshold,
                    overlap_threshold=self.hyper_params.overlap_threshold,
                    center_dist_threshold=self.hyper_params.center_dist_threshold,
                    area_ratio_threshold=self.hyper_params.area_ratio_threshold,
                    pre_nms_topk=self.hyper_params.pre_nms_topk,
                    max_detections=self.hyper_params.max_detections,
                    grid_size=self.hyper_params.grid_size,
                    num_classes=self.hyper_params.classes,
                )
            )
            for batch_idx, image_path in enumerate(image_paths):
                gt_by_image[image_path] = targets[batch_idx]

        metrics = evaluate_map(
            detections_by_image,
            gt_by_image,
            num_classes=self.hyper_params.classes,
            iou_thresholds=self.hyper_params.eval_iou_thresholds,
            labels=self.hyper_params.labels,
            input_dimension=self.hyper_params.network_size,
        )
        current_metric = metrics.get(self.hyper_params.eval_metric_name, 0.0)
        log.info("Eval at batch %d: %s=%.4f", self.batch, self.hyper_params.eval_metric_name, current_metric)
        self.network.save_weights(os.path.join(self.backup_dir, "latest.pt"))

        if self.hyper_params.eval_write_results:
            eval_dir = os.path.join(self.hyper_params.results_dir, f"eval_batch_{self.batch}")
            write_voc_results(
                eval_dir,
                detections_by_image,
                labels=self.hyper_params.labels,
                input_dimension=self.hyper_params.network_size,
            )

        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.network.save_weights(os.path.join(self.backup_dir, "best.pt"))
            log.info("New best checkpoint saved with %s=%.4f", self.hyper_params.eval_metric_name, current_metric)

    def quit(self):
        if self.sigint:
            self._cancel_async_eval()
            self.network.save_weights(os.path.join(self.backup_dir, "backup.pt"))
            return True
        if self.batch >= self.max_batches:
            if self.async_eval_enabled:
                self._wait_for_async_eval()
            self.network.save_weights(os.path.join(self.backup_dir, "final.pt"))
            return True
        return False
