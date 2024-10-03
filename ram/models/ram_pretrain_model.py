import torch
from collections import OrderedDict
import os.path as osp
from tqdm import tqdm

from ram.utils import tensor2img
from ram.utils.registry import MODEL_REGISTRY
from ram.models.mim_util import MaskGenerator, TestMaskGenerator

from .ram_base_model import RAMBaseModel
from ram.metrics import calculate_metric

@MODEL_REGISTRY.register()
class RAMPretrainModel(RAMBaseModel):
    """MIM model for masked image modeling."""

    def __init__(self, opt):
        super(RAMPretrainModel, self).__init__(opt)
    
        # MIM specific settings
        mim_opt = opt.get('mim', None)
        if mim_opt:
            self.setup_mask_generator(mim_opt)

    def setup_mask_generator(self, mim_opt):
        self.mask_patch_size = mim_opt['mask_patch_size']
        self.model_patch_size = mim_opt['model_patch_size']
        self.mask_ratio = mim_opt['mask_ratio']
        self.input_size = self.opt['gt_size']
        self.mask_type = mim_opt.get('mask_type', 'binary')
        
        mask_generator_class = MaskGenerator if self.is_train else TestMaskGenerator
        self.mask_generator = mask_generator_class(
            input_size=self.input_size,
            mask_patch_size=self.mask_patch_size,
            model_patch_size=self.model_patch_size,
            mask_ratio=self.mask_ratio,
            mask_type=self.mask_type
        )

    def feed_data(self, data):
        super(RAMPretrainModel, self).feed_data(data)
        
        if hasattr(self, "mask_generator"):
            b, c, h, w = self.lq.shape
            mask, mask_token = self.mask_generator() if self.is_train else self.mask_generator(h, w)
            
            self.mask = mask.expand(self.gt.shape[0], -1, -1).to(self.device)
            if mask_token is not None:
                self.mask_token = mask_token.expand(self.gt.shape[0], -1, -1, -1).to(self.device)
            else:
                self.mask_token = None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.mask, self.mask_token)
        l_total = 0
        loss_dict = OrderedDict()
        
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt, self.mask)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

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
                self.output_1 = self.net_g(self.lq, self.mask, self.mask_token)
                self.output_2 = self.net_g(self.lq, 1 - self.mask, self.mask_token)
                mask = self.mask.repeat_interleave(1, 1).repeat_interleave(1, 2).unsqueeze(1).contiguous()     
                self.output = self.output_1 * mask + self.output_2 * (1 - mask)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, test_num=-1, save_num=-1):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, test_num, save_num)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, test_num=-1, save_num=-1):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image') if use_pbar else None

        for idx, val_data in enumerate(dataloader):
            if idx == test_num:
                break

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img and idx < save_num:
                self.save_image(current_iter, img_name)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric({'img': sr_img, 'img2': gt_img}, opt_)
            
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if self.gt is not None:
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict


    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)