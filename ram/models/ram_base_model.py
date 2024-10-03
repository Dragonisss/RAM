import os.path as osp
from collections import OrderedDict

import numpy as np
import torch

from .base_model import BaseModel
from ram.utils.dist_util import master_only
from ram.archs import build_network
from ram.losses import build_loss
from ram.utils import get_root_logger, imwrite, tensor2img
class RAMBaseModel(BaseModel):
    """Base model for MIM-related models."""

    def __init__(self, opt):
        super(RAMBaseModel, self).__init__(opt)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        if self.is_train:
            self.init_training_settings()
        else:
            self.load_pretrained_models()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            self.init_ema_model()

        self.load_pretrained_models()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_loss_functions()
        
    def load_pretrained_models(self):
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

    def init_ema_model(self):
        logger = get_root_logger()
        logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
        self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        # 加载预训练模型或初始化权重
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
        else:
            self.model_ema(0)  # 初始化 EMA model 权重
        self.net_g_ema.eval()

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)
        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def setup_loss_functions(self):
        train_opt = self.opt['train']
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        self.gt_path = data['gt_path']
        self.lq = data.get('lq', None)
        if self.lq is not None:
            self.lq = self.lq.to(self.device)
        self.lq_path = data.get('lq_path', None)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_perceptual:
            l_percep = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict


    @master_only
    def save_image(self, current_iter, img_name):
        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        out_img = [sr_img]
        
        if 'mask_img' in visuals:
            mask_img = tensor2img([visuals['mask_img']])
            out_img.append(mask_img)
        if 'gt' in visuals:
            gt_img = tensor2img([visuals['gt']])
            out_img.append(gt_img)
        
        sr_img = np.hstack(out_img)

        # tentative for out of GPU memory
        del self.lq
        del self.output
        torch.cuda.empty_cache()

        if self.opt['is_train']:
            save_img_path = osp.join(self.opt['path']['visualization'],
                                     f'{current_iter}_{img_name}.png')
        else:
            dataset_name = self.opt['datasets']['test']['name']
            if self.opt['val']['suffix']:
                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_{self.opt["val"]["suffix"]}.png')
            else:
                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_{self.opt["name"]}.png')
        
        imwrite(sr_img, save_img_path)
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)