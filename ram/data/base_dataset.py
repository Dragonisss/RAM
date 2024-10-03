from torch.utils import data as data
from torchvision.transforms.functional import normalize

from ram.utils import FileClient, imfrombytes, img2tensor
from ram.data.utils.transforms import augment, paired_random_crop, center_crop

class BaseDataset(data.Dataset):
    def __init__(self, opt):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt.get('io_backend', 'disk')
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.augmentator = None

    def _init_file_client(self):
        if self.file_client is None:
            io_backend_type = self.io_backend_opt['type']
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            self.io_backend_opt['type'] = io_backend_type

    def _load_image(self, path, key):
        img_bytes = self.file_client.get(path, key)
        return imfrombytes(img_bytes, float32=True)

    def _train_augmentation(self, img_gt, img_lq, scale, gt_path=None):
        gt_size = self.opt['gt_size']
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        return img_gt, img_lq

    def _test_processing(self, img_gt, img_lq, scale):
        if self.opt.get('test_crop', False):
            gt_size = self.opt['gt_size']
            img_gt, img_lq = center_crop(img_gt, gt_size), center_crop(img_lq, gt_size)
        img_gt = img_gt[:img_lq.shape[0] * scale, :img_lq.shape[1] * scale, :]
        return img_gt, img_lq

    def _process_images(self, img_gt, img_lq):
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return img_gt, img_lq

    def __getitem__(self, index):
        raise NotImplementedError("Subclass must implement abstract method")

    def __len__(self):
        raise NotImplementedError("Subclass must implement abstract method")