import os
from ram.data.base_dataset import BaseDataset
from ram.utils.registry import DATASET_REGISTRY
from ram.data.utils.data_util import paired_paths_from_folder

@DATASET_REGISTRY.register()
class GoProDataset(BaseDataset):
    def __init__(self, opt, dataroot=None, augmentator=None):
        super(GoProDataset, self).__init__(opt)
        self.folder = dataroot or opt['dataroot']
        self.augmentator = augmentator
        self.paths = self._get_image_paths()

    def _get_image_paths(self):
        paths = []
        for dataset in os.listdir(self.folder):
            lq_folder = os.path.join(self.folder, dataset, 'blur')
            gt_folder = os.path.join(self.folder, dataset, 'sharp')
            paths.extend(paired_paths_from_folder(
                [lq_folder, gt_folder], 
                ['lq', 'gt'], 
                self.opt.get('filename_tmpl', '{}')
            ))
        return paths

    def __getitem__(self, index):
        self._init_file_client()
        scale = self.opt['scale']
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

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