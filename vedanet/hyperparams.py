import logging as log
import os
import torch

__all__ = ['HyperParams']


def _resolve_path(path, base_dir):
    if not path:
        return path
    expanded = os.path.expanduser(os.path.expandvars(path))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(base_dir, expanded))


def _resolve_path_map(path_map, base_dir):
    return {
        os.path.normpath(os.path.expanduser(os.path.expandvars(old_prefix))): _resolve_path(new_prefix, base_dir)
        for old_prefix, new_prefix in path_map.items()
    }

class HyperParams(object):
    def __init__(self, config, train_flag=1):
        
        self.cuda = True
        self.project_root = config.get('project_root', os.getcwd())
        self.labels = config['labels']
        self.classes = len(self.labels)
        self.model_name = config['model_name']
        self.task = config.get('task', 'voc')
        self.data_root = _resolve_path(config.get('data_root_dir', ''), self.project_root)
        self.image_root = _resolve_path(config.get('image_root', ''), self.project_root)
        self.path_remap = _resolve_path_map(config.get('path_remap', {}), self.project_root)
        self.grid_size = config.get('grid_size', 14)
        self.backbone_pretrained = config.get('backbone_pretrained', False)
        self.point_weight = config.get('point_weight', 1.0)
        self.coord_weight = config.get('coord_weight', 2.0)
        self.link_weight = config.get('link_weight', 0.5)
        self.class_weight = config.get('class_weight', 0.5)
        self.eval_iou_thresholds = config.get('eval_iou_thresholds', [0.5, 0.75])
        self.eval_metric_name = config.get('eval_metric_name', 'mAP@0.5')
        self.eval_write_results = config.get('eval_write_results', False)

        # cuda check
        if self.cuda:
            if not torch.cuda.is_available():
                log.debug('CUDA not available')
                self.cuda = False
            else:
                log.debug('CUDA enabled')

        if train_flag == 1:
            cur_cfg = config

            self.nworkers = cur_cfg['nworkers'] 
            self.pin_mem = cur_cfg['pin_mem'] 
            self.network_size = cur_cfg['input_shape']
            self.batch = cur_cfg['batch_size']
            self.mini_batch = cur_cfg['mini_batch_size']
            self.max_batches = cur_cfg['max_batches']
            self.flip = cur_cfg.get('flip', 0.5)
            self.jitter = cur_cfg.get('jitter', 0.3)
            self.crop_min_area = cur_cfg.get('crop_min_area', 0.2)
            self.hue = cur_cfg.get('hue', 0.05)
            self.sat = cur_cfg.get('saturation', 1.5)
            self.val = cur_cfg.get('value', 1.5)
            self.brightness = cur_cfg.get('brightness', 0.15)
            self.contrast = cur_cfg.get('contrast', 0.15)
            self.grayscale = cur_cfg.get('grayscale', 0.0)
            self.blur = cur_cfg.get('blur', 0.0)
            self.noise_std = cur_cfg.get('noise_std', 0.0)

            self.learning_rate = cur_cfg['warmup_lr'] 
            self.max_learning_rate = cur_cfg.get('max_lr', self.learning_rate)
            self.min_learning_rate = cur_cfg.get('min_lr', self.learning_rate)
            self.lr_policy = cur_cfg.get('lr_policy', 'steps')
            self.warmup_batches = cur_cfg.get('warmup_batches', 0)
            self.cosine_total_batches = cur_cfg.get('cosine_total_batches', 0)
            self.backbone_lr_mult = cur_cfg.get('backbone_lr_mult', 1.0)
            self.momentum = cur_cfg['momentum']
            self.decay = cur_cfg['decay'] 
            self.lr_steps = cur_cfg['lr_steps']
            self.lr_rates = cur_cfg['lr_rates'] 

            self.backup = cur_cfg['backup_interval']
            self.bp_steps = cur_cfg['backup_steps']
            self.bp_rates = cur_cfg['backup_rates']
            self.backup_dir = cur_cfg['backup_dir']

            self.resize = cur_cfg['resize_interval'] 
            self.rs_steps = []
            self.rs_rates = []
            self.weights = _resolve_path(cur_cfg['weights'], self.project_root)
            self.clear = cur_cfg['clear']
            if self.task == 'pln':
                self.train_list = _resolve_path(cur_cfg['train_list'], self.project_root)
                self.test_list = _resolve_path(cur_cfg['eval_test_list'], self.project_root)
                self.eval_interval = cur_cfg.get('eval_interval', 0)
                self.eval_batch_size = cur_cfg.get('eval_batch_size', self.batch)
                self.eval_nworkers = cur_cfg.get('eval_nworkers', self.nworkers)
                self.eval_pin_mem = cur_cfg.get('eval_pin_mem', self.pin_mem)
                self.async_eval = cur_cfg.get('async_eval', False)
                self.async_eval_gpus = cur_cfg.get('async_eval_gpus', cur_cfg.get('gpus', '0'))
                self.p_threshold = cur_cfg.get('p_thresh', 0.1)
                self.score_threshold = cur_cfg.get('score_thresh', 0.1)
                self.nms_score = cur_cfg.get('nms_score_thresh', self.score_threshold)
                self.nms_thresh = cur_cfg.get('nms_thresh', 0.5)
                self.min_size = cur_cfg.get('min_size', 0.0)
                self.aspect_ratio_threshold = cur_cfg.get('aspect_ratio_threshold', 100.0)
                self.overlap_threshold = cur_cfg.get('overlap_threshold', 1.5)
                self.center_dist_threshold = cur_cfg.get('center_dist_threshold', 0.0)
                self.area_ratio_threshold = cur_cfg.get('area_ratio_threshold', 0.5)
                self.pre_nms_topk = cur_cfg.get('pre_nms_topk', 512)
                self.max_detections = cur_cfg.get('max_detections', 0)
                self.results_dir = _resolve_path(cur_cfg.get('eval_results', 'results/PLNResnet18'), self.project_root)
            else:
                dataset = cur_cfg['dataset']
                self.trainfile = f'{self.data_root}/{dataset}.pkl'
        elif train_flag == 2:
            cur_cfg = config

            self.nworkers = cur_cfg['nworkers'] 
            self.pin_mem = cur_cfg['pin_mem'] 
            self.network_size = cur_cfg['input_shape']
            self.batch = cur_cfg['batch_size']
            self.weights = _resolve_path(cur_cfg['weights'], self.project_root)
            if self.task == 'pln':
                self.test_list = _resolve_path(cur_cfg['test_list'], self.project_root)
                self.p_threshold = cur_cfg.get('p_thresh', 0.1)
                self.score_threshold = cur_cfg.get('score_thresh', 0.1)
                self.nms_thresh = cur_cfg.get('nms_thresh', 0.5)
                self.nms_score = cur_cfg.get('nms_score_thresh', self.score_threshold)
                self.min_size = cur_cfg.get('min_size', 0.0)
                self.aspect_ratio_threshold = cur_cfg.get('aspect_ratio_threshold', 100.0)
                self.overlap_threshold = cur_cfg.get('overlap_threshold', 1.5)
                self.center_dist_threshold = cur_cfg.get('center_dist_threshold', 0.0)
                self.area_ratio_threshold = cur_cfg.get('area_ratio_threshold', 0.5)
                self.pre_nms_topk = cur_cfg.get('pre_nms_topk', 512)
                self.max_detections = cur_cfg.get('max_detections', 0)
                self.results_dir = _resolve_path(cur_cfg['results'], self.project_root)
            else:
                dataset = cur_cfg['dataset']
                self.testfile = f'{self.data_root}/{dataset}.pkl'
                self.conf_thresh = cur_cfg['conf_thresh']
                self.nms_thresh = cur_cfg['nms_thresh']
                self.results = cur_cfg['results']

        else:
            cur_cfg = config

            self.network_size = cur_cfg['input_shape']
            self.batch = cur_cfg['batch_size']
            self.max_iters = cur_cfg['max_iters']
