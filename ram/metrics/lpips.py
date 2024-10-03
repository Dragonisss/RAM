import numpy as np

from ram.metrics.metric_util import reorder_image, to_y_channel
from ram.utils.registry import METRIC_REGISTRY
from torchvision.transforms.functional import normalize
import lpips
@METRIC_REGISTRY.register()
class calculate_lpips():
    def __init__(self):
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    
    def __call__(self,img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
        """Calculate PSNR (Peak Signal-to-Noise Ratio).

        Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            img (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
            test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

        Returns:
            float: PSNR result.
        """

        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
        if input_order not in ['HWC', 'CHW']:
            raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
        img = reorder_image(img, input_order=input_order)
        img2 = reorder_image(img2, input_order=input_order)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if test_y_channel:
            img = to_y_channel(img)
            img2 = to_y_channel(img2)

        img = img.astype(np.float64)
        img2 = img2.astype(np.float64)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize(img, mean, std, inplace=True)
        normalize(img2, mean, std, inplace=True)

        # calculate lpips
        lpips_val = self.loss_fn_vgg(img.unsqueeze(0).cuda(), img2.unsqueeze(0).cuda())

        return lpips_val
