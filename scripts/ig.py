from os import path as osp
from ram.utils.options import parse_options
from scripts.analysis_utils import BaseAnalysis,attr_grad
from tqdm import tqdm
import numpy as np
import torch

class IGAnalysis(BaseAnalysis):
    def __init__(self, opt):
        super().__init__(opt)

    def analyze(self):
        total_filter_mac = [0.0] * len(self.hook_list)
        for test_loader in self.test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            num_samples = self.opt.get('num_samples',10) 
            print(f'Analyzing {test_set_name}...\n')
            pbar = tqdm(total=num_samples, desc='')
            for idx, val_data in enumerate(test_loader):
                if idx >= num_samples:
                    break
                tensor_lq = val_data['lq'].to(self.device)
                imgname = osp.basename(val_data['lq_path'][0])
                tensor_base = torch.zeros_like(tensor_lq)
                layer_conductance = self._integrated_gradients(tensor_base, tensor_lq)
                total_filter_mac = [a + b for a, b in zip(total_filter_mac, layer_conductance)]
                pbar.set_description(f'Read {imgname}')
                pbar.update(1)
        self._save_results(total_filter_mac, 'ig')

    def _integrated_gradients(self, base_img, final_img):
        total_step = self.opt['total_step']
        order_array = np.random.permutation(final_img.shape[-2] * final_img.shape[-1])
        start_ratio = self.opt['pretrained_ratio']
        all_hook_layer_gradiant = [0.0] * len(self.hook_list)

        for step in range(total_step):
            alpha = 1 - start_ratio + start_ratio * step / total_step
            interpolated_img = self._get_interpolated_img_from_linear_path(base_img, final_img, alpha, order_array)
            self.model.zero_grad()
            interpolated_output = self.model(interpolated_img)
            loss = attr_grad(interpolated_output)
            loss.backward()

            grad = [hook.grad.detach() for hook in self.hook_list]
            all_hook_layer_gradiant = grad if step == 0 else [a + g for a, g in zip(all_hook_layer_gradiant, grad)]

        return [torch.mean(torch.abs(ig)).detach().cpu().numpy() for ig in all_hook_layer_gradiant]

def main():
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt, _ = parse_options(root_path, is_train=False)

    analysis = IGAnalysis(opt)
    analysis.analyze()

if __name__ == '__main__':
    main()