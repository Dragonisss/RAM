import os.path as osp
from ram.utils.options import parse_options
from scripts.analysis_utils import BaseAnalysis
import torch
import numpy as np
import os


class RandomAnalysis(BaseAnalysis):
    def __init__(self, opt):
        super().__init__(opt)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def analyze(self):
        name_list = [hook.name for hook in self.hook_list]
        np.random.shuffle(name_list)
        
        save_folder = f"options/cond/{self.opt['name']}"
        os.makedirs(save_folder, exist_ok=True)
        np.savetxt(os.path.join(save_folder, 'filter_name.txt'), name_list, delimiter=',', fmt='%s')

def main():
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt, _ = parse_options(root_path, is_train=False)
    
    analysis = RandomAnalysis(opt)
    analysis.analyze()

if __name__ == '__main__':
    main()