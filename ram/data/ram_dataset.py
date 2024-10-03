from torch.utils import data as data
from ram.data.paired_image_dataset import PairedImageDataset
from ram.data.dehaze_dataset import DehazeDataset
from ram.data.gopro_dataset import GoProDataset
from ram.data.low_cost_dataset import LowCostDataset
from ram.data.lolv2_dataset import LOLv2Dataset
from ram.utils.registry import DATASET_REGISTRY
from ram.data.utils.online_util import parse_degradations
import random

@DATASET_REGISTRY.register()
class RAMDataset(data.Dataset):
    def __init__(self, opt):
        super(RAMDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = None,None #opt['dataroot_gt'], opt['dataroot_lq']
        self.ots_dataset = DehazeDataset(opt,lq_path=opt['ots_lq_path'],gt_path=opt['ots_gt_path'])
        self.rain13k_dataset = PairedImageDataset(opt,lq_path=opt['rain_lq_path'],gt_path=opt['rain_gt_path'])
        self.deblur_dataset = GoProDataset(opt,dataroot=opt['gopro_path'])
        self.lol_dataset = LOLv2Dataset(opt,dataroot=opt['lol_v2_path'])

        augmentators = parse_degradations(opt['augment'])
        low_cost_datasets = [LowCostDataset(opt,dataroot=opt['lsdir_path'],augmentator=augmentator) for augmentator in augmentators]
        high_cost_datasets = [self.ots_dataset,self.rain13k_dataset,self.deblur_dataset,self.lol_dataset]

        self.datasets = high_cost_datasets + low_cost_datasets
        self.types = len(self.datasets)
        self.ids = [0] * self.types
        self.step = 0

    def __getitem__(self, index):
        type_ids = index % self.types
        # print("type_ids:", type_ids)
        sample = self.datasets[type_ids][self.ids[type_ids]]
        self.ids[type_ids] = (self.ids[type_ids] + 1 ) % len(self.datasets[type_ids])
        if self.ids[type_ids] == 0:
            random.shuffle(self.datasets[type_ids].paths)

        return sample

    def __len__(self):
        length = min([len(d) for d in self.datasets])
        # print(length)
        return length * self.types * 10
