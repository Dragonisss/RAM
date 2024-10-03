import os
from ram.data.base_dataset import BaseDataset
from ram.utils.registry import DATASET_REGISTRY
from ram.data.utils.transforms import augment, random_crop, center_crop
from ram.utils import imfrombytes
from ram.data.utils.online_util import parse_degradations,AugmentatorHub


@DATASET_REGISTRY.register()
class LowCostDataset(BaseDataset):
    def __init__(self, opt, dataroot=None, augmentator=None):
        super(LowCostDataset, self).__init__(opt)
        self.gt_folder = dataroot or opt['dataroot_gt']
        self.paths = self._get_image_paths()
        self.augmentator = self._init_augmentator(augmentator)

    def _get_image_paths(self):
        image_paths = []
        dataset_list = os.listdir(self.gt_folder)
        for subset in sorted([s for s in dataset_list if s[:2] == '00' or not self.opt['is_train']]):
            subset_path = os.path.join(self.gt_folder, subset)
            image_paths.extend([os.path.join(subset_path, img) for img in os.listdir(subset_path)])
        return sorted(image_paths)

    def _init_augmentator(self, augmentator):
        if augmentator is None:
            augmentators = parse_degradations(self.opt['augment'])
            return augmentators[0] if len(augmentators) == 1 else AugmentatorHub(augmentators)
        return augmentator

    def __getitem__(self, index):
        self._init_file_client()
        
        gt_path = self.paths[index]
        img_gt = self._load_image(gt_path)
        
        if self.opt['is_train']:
            img_gt = self._train_preprocessing(img_gt)
        else:
            img_gt = self._test_preprocessing(img_gt)
        
        img_lq = self.augmentator(img_gt)
        
        img_gt, img_lq = self._process_images(img_gt, img_lq)
        
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': gt_path, 'gt_path': gt_path}

    def _load_image(self, path):
        img_bytes = self.file_client.get(path)
        return imfrombytes(img_bytes, float32=True)

    def _train_preprocessing(self, img):
        img = augment(img, hflip=self.opt.get('use_hflip', False), rotation=False)
        return random_crop(img, self.opt['gt_size'])

    def _test_preprocessing(self, img):
        if self.opt.get('test_crop', False):
            return center_crop(img, self.opt['gt_size'])
        return img

    def __len__(self):
        return len(self.paths)
