import numpy as np
import random
import torch
from collections import OrderedDict
from torch.nn import functional as F

from basicsr.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from basicsr.data.transforms import paired_random_crop, paired_random_crop_seg
from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register(suffix="basicsr")
class PhenoSRGANModel(SRGANModel):
    def __init__(self, opt):
        super(PhenoSRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(
            differentiable=False
        ).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get("queue_size", 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, "queue_lr"):
            assert (
                self.queue_size % b == 0
            ), f"queue size {self.queue_size} should be divisible by batch size {b}"
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first batch samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()

            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue

        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.lq.clone()
            )
            self.queue_gt[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.gt.clone()
            )
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images."""
        if self.is_train and self.opt.get("high_order_degradation", True):
            # training data synthesis
            self.gt = data["gt"].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)
            self.kernel1 = data["kernel1"].to(self.device)
            self.kernel2 = data["kernel2"].to(self.device)
            self.sinc_kernel = data["sinc_kernel"].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(
                ["up", "down", "keep"], self.opt["resize_prob"]
            )[0]
            if updown_type == "up":
                scale = np.random.uniform(1, self.opt["resize_range"][1])
            elif updown_type == "down":
                scale = np.random.uniform(self.opt["resize_range"][0], 1)
            else:
                scale = 1
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt["gray_noise_prob"]
            if np.random.uniform() < self.opt["gaussian_noise_prob"]:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range"],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range"],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range"])
            out = torch.clamp(
                out, 0, 1
            )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt["second_blur_prob"]:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(
                ["up", "down", "keep"], self.opt["resize_prob2"]
            )[0]
            if updown_type == "up":
                scale = np.random.uniform(1, self.opt["resize_range2"][1])
            elif updown_type == "down":
                scale = np.random.uniform(self.opt["resize_range2"][0], 1)
            else:
                scale = 1
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(
                out,
                size=(
                    int(ori_h / self.opt["scale"] * scale),
                    int(ori_w / self.opt["scale"] * scale),
                ),
                mode=mode,
            )
            # add noise
            gray_noise_prob = self.opt["gray_noise_prob2"]
            if np.random.uniform() < self.opt["gaussian_noise_prob2"]:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range2"],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range2"],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )

            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(
                    out,
                    size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                    mode=mode,
                )
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(
                    out,
                    size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                    mode=mode,
                )
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

            # random crop
            gt_size = self.opt["gt_size"]
            (self.gt, self.gt_usm), self.lq = paired_random_crop(
                [self.gt, self.gt_usm], self.lq, gt_size, self.opt["scale"]
            )
            self._dequeue_and_enqueue()
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()
        else:
            # for paired training or validation
            self.lq = data["lq"].to(self.device)
            if "gt" in data:
                self.gt = data["gt"].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(PhenoSRGANModel, self).nondist_validation(
            dataloader, current_iter, tb_logger, save_img
        )
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt["l1_gt_usm"] is False:
            l1_gt = self.gt
        if self.opt["percep_gt_usm"] is False:
            percep_gt = self.gt
        if self.opt["gan_gt_usm"] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d_coarse.parameters():
            p.requires_grad = False

        for p in self.net_d_refine.parameters():
            p.requires_grad = False

        self.optimizer_g_coarse.zero_grad()
        self.optimizer_g_refine.zero_grad()

        self.output = self.net_g(self.lq, current_iter)
        if self.cri_ldl:
            self.output_ema = self.net_g_ema(self.lq)

        l_g_total_coarse = 0
        l_g_total_refine = 0
        loss_dict = OrderedDict()
        if (
            current_iter % self.net_d_iters == 0
            and current_iter > self.net_d_init_iters
        ):
            if self.cri_pix:
                l_g_pix_coarse = self.cri_pix(
                    pred=self.output, target=l1_gt, stage=self.opt["stage"]
                )
                l_g_total_coarse += l_g_pix_coarse
                loss_dict["l_g_pix_coarse"] = l_g_pix_coarse

                l_g_pix_refine = self.cri_pix(
                    pred=self.output, target=l1_gt, stage=self.opt["stage"]
                )
                l_g_total_refine += l_g_pix_refine
                loss_dict["l_g_pix_refine"] = l_g_pix_refine

            if self.cri_perceptual:
                l_g_percep_coarse, l_g_style_coarse = self.cri_perceptual(
                    self.output[0], percep_gt
                )
                if l_g_percep_coarse is not None:
                    l_g_total_coarse += l_g_percep_coarse
                    loss_dict["l_g_percep_coarse"] = l_g_percep_coarse
                if l_g_style_coarse is not None:
                    l_g_total_coarse += l_g_style_coarse
                    loss_dict["l_g_style"] = l_g_style_coarse

                l_g_percep_refine, l_g_style_refine = self.cri_perceptual(
                    self.output[1], percep_gt
                )
                if l_g_percep_refine is not None:
                    l_g_total_refine += l_g_percep_refine
                    loss_dict["l_g_percep_refine"] = l_g_percep_refine
                if l_g_style_refine is not None:
                    l_g_total_refine += l_g_style_refine
                    loss_dict["l_g_style"] = l_g_style_refine

            fake_g_pred_coarse = self.net_d_coarse(self.output[0])
            l_g_gan_coarse = self.cri_gan(fake_g_pred_coarse, True, is_disc=False)
            l_g_total_coarse += l_g_gan_coarse
            loss_dict["l_g_gan_coarse"] = l_g_gan_coarse

            l_g_total_coarse.backward(retain_graph=True)

            fake_g_pred_refine = self.net_d_refine(self.output[1])
            l_g_gan_refine = self.cri_gan(fake_g_pred_refine, True, is_disc=False)
            l_g_total_refine += l_g_gan_refine
            loss_dict["l_g_gan_refine"] = l_g_gan_refine

            l_g_total_refine.backward()

            self.optimizer_g_coarse.step()
            self.optimizer_g_refine.step()

        for p in self.net_d_coarse.parameters():
            p.requires_grad = True

        for p in self.net_d_refine.parameters():
            p.requires_grad = True

        self.optimizer_d_coarse.zero_grad()
        self.optimizer_d_refine.zero_grad()

        real_d_pred_coarse = self.net_d_coarse(gan_gt)
        l_d_real_coarse = self.cri_gan(real_d_pred_coarse, True, is_disc=True)
        loss_dict["l_d_real_coarse"] = l_d_real_coarse
        loss_dict["out_d_real_coarse"] = torch.mean(real_d_pred_coarse.detach())
        l_d_real_coarse.backward()

        fake_d_pred_coarse = self.net_d_coarse(
            self.output[0].detach().clone()
        )  # clone for pt1.9
        l_d_fake_coarse = self.cri_gan(fake_d_pred_coarse, False, is_disc=True)
        loss_dict["l_d_fake_coarse"] = l_d_fake_coarse
        loss_dict["out_d_fake_coarse"] = torch.mean(fake_d_pred_coarse.detach())
        l_d_fake_coarse.backward()
        self.optimizer_d_coarse.step()

        real_d_pred_refine = self.net_d_refine(gan_gt)
        l_d_real_refine = self.cri_gan(real_d_pred_refine, True, is_disc=True)
        loss_dict["l_d_real_refine"] = l_d_real_refine
        loss_dict["out_d_real_refine"] = torch.mean(real_d_pred_refine.detach())
        l_d_real_refine.backward()

        fake_d_pred_refine = self.net_d_refine(
            self.output[1].detach().clone()
        )  # clone for pt1.9
        l_d_fake_refine = self.cri_gan(fake_d_pred_refine, False, is_disc=True)
        loss_dict["l_d_fake_refine"] = l_d_fake_refine
        loss_dict["out_d_fake_refine"] = torch.mean(fake_d_pred_refine.detach())
        l_d_fake_refine.backward()
        self.optimizer_d_refine.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
