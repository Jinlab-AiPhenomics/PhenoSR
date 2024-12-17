import numpy as np
import random
from os import path as osp
from tqdm import tqdm
import torch
import cv2
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.metrics import calculate_metric

@MODEL_REGISTRY.register(suffix='basicsr')
class HRNetModel(SRModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(HRNetModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.origin.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            _, c, h, w = self.origin.size()
            # 语义分割jpg
            self.queue_origin = torch.zeros(self.queue_size, c, h, w).cuda()
            # 语义分割标签
            _, h, w = self.label.size()
            self.queue_label = torch.zeros(self.queue_size, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle 随机打乱
            idx = torch.randperm(self.queue_size)
            self.queue_origin = self.queue_origin[idx]
            self.queue_label = self.queue_label[idx]
            # get first b samples
            origin_dequeue = self.queue_origin[0:b, :, :, :].clone()
            label_dequeue = self.queue_label[0:b, :, :].clone()
            # update the queue
            self.queue_origin[0:b, :, :, :] = self.origin.clone()
            self.queue_label[0:b, :, :] = self.label.clone()

            self.origin = origin_dequeue
            self.label = label_dequeue
        else:
            # only do enqueue
            self.queue_origin[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.origin.clone()
            self.queue_label[self.queue_ptr:self.queue_ptr + b, :, :] = self.label.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.origin = data['origin'].to(self.device)
            self.label = data['label'].to(self.device)
            # training pair pool
            self._dequeue_and_enqueue()
        else:
            self.origin = data['origin'].to(self.device)
            self.label = data['label'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.cri_seg:
            self.seg_res = self.net_g(self.origin)

        l_total = 0
        loss_dict = OrderedDict()
        # seg loss
        if self.cri_seg:
            l_seg = self.cri_seg(self.seg_res, self.label)
            l_total += l_seg
            loss_dict['l_seg'] = l_seg

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.seg_res = self.net_g_ema(self.origin)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.seg_res = self.net_g(self.origin)
            self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train = False
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        self.val_loss = 0
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['seg_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            seg_img = torch.argmax(visuals['seg_result'], dim=1, keepdim=True).squeeze().cpu().numpy()
            metric_data['img3'] = seg_img
            metric_data['img'] = seg_img
            if 'label' in visuals:
                label = visuals['label']
                # 计算miou
                metric_data['img4'] = label.squeeze(0).detach().cpu().numpy()
                metric_data['img2'] = label.squeeze(0).detach().cpu().numpy()
                # 计算验证损失
                l_seg = self.cri_seg(visuals['seg_result'].cuda(), label.cuda())
                self.val_loss += l_seg
                del self.label
            # tentative for out of GPU memory
            del self.seg_res
            torch.cuda.empty_cache()

            if save_img:
                save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                         f'{img_name}_{current_iter}_seg.png')
                seg_map = torch.argmax(visuals['seg_result'], dim=1, keepdim=True).squeeze().cpu().numpy()
                imwrite(seg_map * 255, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] = self.metric_results[metric].astype(np.float64)
                self.metric_results[metric] /= np.float64((idx + 1))
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        if tb_logger:
            tb_logger.add_scalar(f'losses/{dataset_name}/l_seg_val', self.val_loss / (idx + 1), current_iter)

        self.is_train = True

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # 分割结果
        out_dict['seg_result'] = self.seg_res.detach().cpu()
        # 分割标签
        if hasattr(self, 'label'):
            out_dict['label'] = self.label
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
