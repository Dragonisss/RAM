import os
from ram.data.base_dataset import BaseDataset
from ram.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class DehazeDataset(BaseDataset):
    def __init__(self, opt, lq_path=None, gt_path=None, augmentator=None):
        super(DehazeDataset, self).__init__(opt)
        self.gt_folder = gt_path or opt['dataroot_gt']
        self.lq_folder = lq_path or opt['dataroot_lq']
        self.augmentator = augmentator
        self.paths = os.listdir(self.lq_folder)

    def __getitem__(self, index):
        self._init_file_client()

        scale = self.opt.get('scale', 1)
        gt_path = self._get_gt_path(index)
        lq_path = os.path.join(self.lq_folder, self.paths[index])

        img_gt = self._load_image(gt_path, 'gt')
        img_lq = self._load_image(lq_path, 'lq')

        if self.opt['phase'] == 'train':
            img_gt, img_lq = self._train_augmentation(img_gt, img_lq, scale, gt_path)
        else:
            img_gt, img_lq = self._test_processing(img_gt, img_lq, scale)

        if self.augmentator:
            img_lq = self.augmentator(img_lq)

        img_gt, img_lq = self._process_images(img_gt, img_lq)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

    def _get_gt_path(self, index):
        if 'SOTS' in self.opt['name']:
            return os.path.join(self.gt_folder, f"{self.paths[index].split('_')[0]}.png")
        return os.path.join(self.gt_folder, f"{self.paths[index].split('_')[0]}.jpg")